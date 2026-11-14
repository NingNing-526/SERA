"""Dataset dispatch and the legacy similarity-matrix refresh helper."""

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


def _build_eval_dataloader(dataset, split, batch_size, config):
    """Build a deterministic validation/test loader once and reuse it."""
    if dataset == 'mosi':
        return MOSIDataloader(
            split, shuffle=False, num_workers=0,
            batch_size=batch_size, config=config
        )
    if dataset == 'mosei':
        return MOSEIDataloader(
            split, shuffle=False, num_workers=0,
            batch_size=batch_size, config=config
        )
    if dataset == 'sims':
        return SIMSDataloader(
            split, shuffle=False, num_workers=0, batch_size=batch_size
        )
    raise ValueError(f'Unsupported dataset: {dataset!r}')


def update_matrix(data, model, dataset, config=default_config):
    with torch.no_grad():
        model.eval()
        if dataset == 'mosi':
            train_data = MOSIDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False, 
                                    shuffle=False,use_sampler=False, config=config)
        elif dataset == 'mosei':
            train_data = MOSEIDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False,
                                     shuffle=False, use_sampler=False, config=config)
        elif dataset == 'sims':
            train_data = SIMSDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False,
                                     shuffle=False, use_sampler=False)
        
        device = config.DEVICE
        print('Collecting New embeddings')
        T, V, A = [], [], []
        bar = tqdm(train_data, disable=True)
        for index, sample in enumerate(bar):
            _T, _V, _A = model(sample, None, return_loss=False)
            T.append(_T.detach())
            V.append(_V.detach())
            A.append(_A.detach())
        T = torch.cat(T, dim=0).to(torch.device('cpu')).squeeze()
        V = torch.cat(V, dim=0).to(torch.device('cpu')).squeeze()
        A = torch.cat(A, dim=0).to(torch.device('cpu')).squeeze()
        print('Updating Similarity Matrix')
        data.dataset.update_matrix(T, V, A)
        model.train()
    return
