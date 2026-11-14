"""Training loop with unchanged parameter groups, losses, schedules and early stop."""

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
from engine.evaluation import eval


def SERA_train(dataset=None, check=None, config=default_config):
    print('---------------SERA_EXP_---------------')
    if check is None:
        check = {'Loss': 10000, 'MAE': 100}
    else:
        check = check.copy()
    metrics = Metrics()
    train_bool = [True, True, True, True, True]

    model = SERA(config=config)

    model.set_train(train_bool)

    device = config.DEVICE
    dataset = config.dataset
    batch_size = config.PARAM.Train.batch_size
    lr = config.PARAM.Train.lr
    encoder_lr = getattr(config.PARAM.Train, 'encoder_lr', lr)
    new_module_lr = getattr(config.PARAM.Train, 'new_module_lr', lr)
    total_epoch = config.PARAM.Train.epoch
    decay = config.PARAM.Train.decay
    num_warm_up = config.PARAM.Train.num_warm_up

    if dataset == 'mosi':
        train_data = MOSIDataloader('train', batch_size=batch_size, use_similarity=False, simi_return_mono=False, use_sampler=False, config=config)
        log_path = config.MOSI.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    elif dataset == 'mosei':
        train_data = MOSEIDataloader('train', batch_size=batch_size, use_similarity=False, simi_return_mono=False, use_sampler=False, config=config)
        log_path = config.MOSEI.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    elif dataset == 'sims':
        train_data = SIMSDataloader('train', batch_size=batch_size, use_similarity=False, simi_return_mono=False, use_sampler=False)
        log_path = config.SIMS.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    else:
        raise ValueError(f'Unsupported dataset: {dataset!r}')

    # Construct validation data outside the epoch loop.  Recreating it every
    # epoch previously reloaded the multi-gigabyte MOSEI semantic file at each
    # "EVAL valid", making validation look as if it had stalled.
    valid_data = _build_eval_dataloader(
        dataset, 'valid', batch_size=batch_size, config=config
    )
    check_dir(log_path.rsplit('/', 1)[0])

    run_config_log = (
        f'Run config: dataset={dataset}, '
        f'variant={config.PARAM.downStream.experiment_variant}, '
        f'seed={config.seed}, '
        f'batch_size={batch_size}, epochs={total_epoch}, '
        f'encoder_lr={encoder_lr:g}, new_module_lr={new_module_lr:g}, '
        f'weight_decay={decay:g}, warmup_epochs={num_warm_up}, '
        f'dropout={config.PARAM.downStream.drop_out:g}, '
        f'top_k(text/audio/vision)='
        f'{getattr(config.PARAM.downStream, "semantic_text_top_k", config.PARAM.downStream.semantic_top_k)}/'
        f'{getattr(config.PARAM.downStream, "semantic_audio_top_k", config.PARAM.downStream.semantic_top_k)}/'
        f'{getattr(config.PARAM.downStream, "semantic_vision_top_k", config.PARAM.downStream.semantic_top_k)}, '
        f'gated_align={getattr(config.PARAM.downStream, "semantic_alignment_gated", False)}, '
        f'align_gate_init={getattr(config.PARAM.downStream, "semantic_alignment_gate_init", 0.1):g}, '
        f'text_modality_gate={getattr(config.PARAM.downStream, "text_guided_modality_gate", False)}, '
        f'no_tgf_mode={getattr(config.PARAM.downStream, "no_tgf_fusion_mode", "legacy_concat")}, '
        f'aux_start={config.PARAM.downStream.semantic_aux_start_weight:g}, '
        f'aux_min={config.PARAM.downStream.semantic_aux_min_weight:g}, '
        f'early_stop(metric/patience/min_epochs/min_delta)='
        f'{config.PARAM.Train.early_stopping_metric}/'
        f'{config.PARAM.Train.early_stopping_patience}/'
        f'{config.PARAM.Train.early_stopping_min_epochs}/'
        f'{config.PARAM.Train.early_stopping_min_delta:g}'
    )
    print(run_config_log)
    write_log(run_config_log, path=log_path)
    
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    encoder_prefixes = ('text_encoder.', 'vision_encoder.', 'audio_encoder.')
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    optimizer_grouped_parameters = []
    for group_name, is_encoder, group_lr in (
        ('encoder', True, encoder_lr),
        ('new', False, new_module_lr),
    ):
        for decay_name, apply_decay in (('decay', True), ('no_decay', False)):
            parameters = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(encoder_prefixes) == is_encoder
                and (not any(token in name for token in no_decay)) == apply_decay
            ]
            if parameters:
                optimizer_grouped_parameters.append({
                    "params": parameters,
                    "lr": group_lr,
                    "weight_decay": decay if apply_decay else 0.0,
                    "group_name": f'{group_name}_{decay_name}',
                })

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=new_module_lr,
        amsgrad=False,
    )
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        num_warm_up * (len(train_data))),num_training_steps=total_epoch * len(train_data),)
    model.to(device)


    loss = 0
    all_loss = 0
    aux_loss = 0
    early_stop_metric = config.PARAM.Train.early_stopping_metric
    early_stop_patience = config.PARAM.Train.early_stopping_patience
    early_stop_min_epochs = config.PARAM.Train.early_stopping_min_epochs
    early_stop_min_delta = config.PARAM.Train.early_stopping_min_delta
    early_stop_minimize = early_stop_metric in {'MAE', 'Loss'}
    early_stop_best = float('inf') if early_stop_minimize else -float('inf')
    early_stop_bad_epochs = 0
    best_validation = {}

    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_data, disable=False)
        aux_start_weight = getattr(
            config.PARAM.downStream, 'semantic_aux_start_weight', 1.0
        )
        aux_progress = (epoch - 1) / max(total_epoch - 1, 1)
        selector_enabled = (
            config.PARAM.downStream.use_gemma_semantic
            and getattr(
                config.PARAM.downStream, 'semantic_selection', True
            )
        )
        auxiliary_enabled = (
            selector_enabled
            and getattr(
                config.PARAM.downStream, 'semantic_auxiliary', True
            )
        )
        aux_weight = max(
            config.PARAM.downStream.semantic_aux_min_weight,
            aux_start_weight * (1.0 - aux_progress),
        ) if auxiliary_enabled else 0.0
        epoch_total_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_sample_count = 0
        
        for index, sample1, in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|All_loss:%s|Loss:%s|Aux_loss:%s|Aux_w:%.3f" % (
                    epoch, all_loss.item(), loss.item(), aux_loss.item(), aux_weight))
            except:
                bar.set_description(
                    "Epoch:%d|All_loss:%.6f|Loss:%.6f|Aux_loss:%.6f|Aux_w:%.3f" % (
                        epoch, all_loss, loss, aux_loss, aux_weight))

            optimizer.zero_grad()

            pred, all_loss, loss, aux_loss = model(
                sample1, None, return_loss=True, aux_weight=aux_weight
            )
            
            all_loss.backward()
            grad_clip_norm = getattr(config.PARAM.Train, 'grad_clip_norm', None)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            current_batch = sample1['regression_labels'].numel()
            epoch_total_loss += all_loss.item() * current_batch
            epoch_pred_loss += loss.item() * current_batch
            epoch_aux_loss += aux_loss.item() * current_batch
            epoch_sample_count += current_batch

        current_lrs = {
            group['group_name']: group['lr']
            for group in optimizer.param_groups
        }
        train_log = (
            f'Train Epoch{epoch}: total={epoch_total_loss / epoch_sample_count:.6f}, '
            f'pred={epoch_pred_loss / epoch_sample_count:.6f}, '
            f'aux={epoch_aux_loss / epoch_sample_count:.6f}, '
            f'aux_weight={aux_weight:.4f}, '
            f'encoder_lr={current_lrs.get("encoder_decay", 0.0):.8g}, '
            f'new_lr={current_lrs.get("new_decay", 0.0):.8g}'
        )
        print(train_log)
        write_log(train_log, path=log_path)

        print("EVAL valid")
        result, result_loss = eval(
            model, dataset, metrics, valid_data, device, config
        )
        if dataset=='sims':
            log1 = 'Valid_Metric Epoch:%d\n\tacc_2:%s\n\tF1_score:%s\n\tacc_3"%s\n\t' \
                'acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '---------------------------------------' % (epoch, 
                result['Mult_acc_2'], result['F1_score'], result['Mult_acc_3'], 
                result['Mult_acc_5'], result['MAE'], result['Corr'], result_loss)     

        else :
            log1 = 'Valid_Metric Epoch%s:\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
                'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '-----------------------------------------' % (epoch,
                result['Has0_acc_2'],result['Has0_F1_score'],result['Non0_acc_2'], result['Non0_F1_score'],
                result['Mult_acc_5'],result['Mult_acc_7'], result['MAE'],result['Corr'],result_loss)   
        
        print(log1)
        write_log(log1, path=log_path)

        # Compare and save from the first validation epoch. Skipping epoch 1
        # would leave ``check`` at its sentinel value, causing epoch 2 to be
        # saved even when it is worse than epoch 1.
        for key, previous_best in check.items():
            current_value = float(result[key])
            improved = (
                current_value <= previous_best
                if key in {'MAE', 'Loss'}
                else current_value >= previous_best
            )
            if improved:
                best_validation[key] = {
                    'epoch': epoch,
                    'metrics': {
                        metric_name: float(metric_value)
                        for metric_name, metric_value in result.items()
                    },
                }
        check = check_and_save(model, dataset, result, check, save_model=True)

        if early_stop_patience > 0:
            if early_stop_metric not in result:
                raise KeyError(
                    f'Early-stopping metric {early_stop_metric!r} is absent '
                    f'from validation results: {sorted(result)}'
                )
            current_metric = float(result[early_stop_metric])
            if early_stop_minimize:
                improved = current_metric < (
                    early_stop_best - early_stop_min_delta
                )
            else:
                improved = current_metric > (
                    early_stop_best + early_stop_min_delta
                )

            if improved:
                early_stop_best = current_metric
                early_stop_bad_epochs = 0
            elif epoch >= early_stop_min_epochs:
                early_stop_bad_epochs += 1
            else:
                # Observe the metric and update the best value during the
                # mandatory warm-up period, but start patience counting only
                # after the requested minimum number of epochs.
                early_stop_bad_epochs = 0

            early_stop_log = (
                f'EarlyStop Epoch{epoch}: metric={early_stop_metric}, '
                f'current={current_metric:.6f}, best={early_stop_best:.6f}, '
                f'bad_epochs={early_stop_bad_epochs}/{early_stop_patience}'
            )
            print(early_stop_log)
            write_log(early_stop_log, path=log_path)

            if (
                epoch >= early_stop_min_epochs
                and early_stop_bad_epochs >= early_stop_patience
            ):
                stop_log = (
                    f'Early stopping at epoch {epoch}: '
                    f'{early_stop_metric} did not improve by more than '
                    f'{early_stop_min_delta:g} for '
                    f'{early_stop_patience} consecutive epochs. '
                    f'Best={early_stop_best:.6f}.'
                )
                print(stop_log)
                write_log(stop_log, path=log_path)
                break

    return best_validation
