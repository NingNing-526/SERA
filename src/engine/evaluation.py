"""Validation/test inference and metric reporting. Existing API names retained."""

import torch
from tqdm import tqdm
import transformers as trans
import datetime
import numpy as np
import config as default_config
from model.sera import SERA
from dataloader.MOSI import MOSIDataloader
from dataloader.MOSEI import MOSEIDataloader
from dataloader.SIMS import SIMSDataloader
from utils import write_log, check_and_save, check_dir, Metrics
from engine.data import _build_eval_dataloader


def eval(model, dataset=None, metrics=None, eval_data=None, device=None, config=default_config):
    batch_size = config.PARAM.Train.batch_size
    with torch.no_grad():
        model.eval()
        if device is None: device = config.DEVICE
        if eval_data is None:
            eval_data = _build_eval_dataloader(
                dataset, 'test', batch_size=batch_size, config=config
            )
        elif isinstance(eval_data, str):
            eval_data = _build_eval_dataloader(
                dataset, eval_data, batch_size=batch_size, config=config
            )
        if metrics is None: metrics = Metrics()
        pred = []
        truth = []
        loss = 0
        sample_count = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            label = sample['regression_labels'].clone().detach().to(device).float()
            _pred, _all_loss, _loss, _, = model(
                sample, None, return_loss=True, aux_weight=0.0
            )
            pred.append(_pred.view(-1))
            truth.append(label)
            current_batch = label.numel()
            loss += _loss.item() * current_batch
            sample_count += current_batch
        
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        if dataset =='sims':
            eval_results = metrics.eval_sims_regression(truth, pred)
        else:
            eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / sample_count
        model.train()

    return eval_results, loss / sample_count


def SERA_test(dataset=None, check_list=None, config=default_config):
    if check_list is None: check_list = ['Has0_F1_score', 'MAE']
    if not isinstance(check_list, list): check_list = [check_list]
    seed = config.seed

    model = SERA(config=config)

    device = config.DEVICE
    model.to(device)
    test_data = _build_eval_dataloader(
        dataset,
        'test',
        batch_size=config.PARAM.Train.batch_size,
        config=config,
    )
    check = {}
    result = None
    print('Evaluating model')
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name='best_'+metric, dataset=dataset)
        result, loss = eval(
            model=model,
            dataset=dataset,
            eval_data=test_data,
            device=device,
            config=config,
        )
        check[metric] = {}
        for key in result.keys():
            check[metric][key] = result[key]

        if dataset=='sims':
            log = 'Test_Metric\n\tacc_2:%s\n\tF1_score:%s\n\tacc_3:%s\n\t' \
                'acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '------------------------------------------' % (result['Mult_acc_2'], result['F1_score'],
                                                                result['Mult_acc_3'], result['Mult_acc_5'],
                                                                result['MAE'], result['Corr'], loss)                                                                                                   

        else:
            log = 'Test_Metric\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2:%s\n\t' \
                'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '------------------------------------------' % (result['Has0_acc_2'], result['Has0_F1_score'],
                                                                result['Non0_acc_2'], result['Non0_F1_score'],
                                                                result['Mult_acc_5'], result['Mult_acc_7'],
                                                                result['MAE'], result['Corr'], loss)
                                                                
        print(log)

    return check
