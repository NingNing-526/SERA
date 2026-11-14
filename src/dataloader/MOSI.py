"""MOSI loading: raw features, exact semantic ID alignment, valid lengths and masks."""

import torch
import numpy as np
import random
import _pickle as pk
import os
import config as default_config
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import samplers
from dataloader.mask_utils import lengths_to_padding_mask


class MOSIDataset(Dataset):
    def __init__(self, type, use_similarity=False, simi_return_mono=False, config=default_config):
        raw_data_path = config.MOSI.raw_data_path
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[type]
        self._load_offline_semantics(type, config)
        # print(self.data.keys())
        #print(self.data['vision'][0])
        if 'audio_lengths' not in self.data.keys():
            audio_len = self.data['audio'].shape[1]
            self.data['audio_lengths'] = [audio_len] * self.data['audio'].shape[0]
        if 'vision_lengths' not in self.data.keys():
            vision_len = self.data['vision'].shape[1]
            self.data['vision_lengths'] = [vision_len] * self.data['vision'].shape[0]

        self.simi_return_mono = simi_return_mono
        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.data['id'] = np.array(self.data['id'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
        #print(self.vision_fea_size)
        self.scaled_embedding_averaged = False
        self.__gen_mask()
        if type == 'train' and use_similarity:
            self.__scale()
            self.__gen_cos_matrix()
        self.use_similarity = use_similarity

    def _load_offline_semantics(self, split, config):
        """Attach fixed-length token features generated before training."""
        if not config.PARAM.downStream.use_gemma_semantic:
            return

        semantic_path = config.MOSI.semantic_feature_path
        if not os.path.isfile(semantic_path):
            if config.PARAM.downStream.semantic_required:
                raise FileNotFoundError(
                    f"Offline semantic feature file not found: {semantic_path}. "
                    "Run scripts/precompute_semantic_tokens.py before training, "
                    "or set PARAM.downStream.use_gemma_semantic=False."
                )
            return

        semantic_data = torch.load(semantic_path, map_location='cpu')
        if split not in semantic_data:
            raise KeyError(f"Split '{split}' is missing from {semantic_path}")
        split_data = semantic_data[split]

        expected_size = len(self.data['raw_text'])
        if 'ids' not in split_data:
            raise KeyError(
                f"'{split}.ids' is missing from {semantic_path}. Regenerate "
                'the offline semantic feature file with the current '
                'precompute_semantic_tokens.py so sample order can be verified.'
            )
        raw_ids = [str(item) for item in self.data['id']]
        semantic_ids = [str(item) for item in split_data['ids']]
        if len(semantic_ids) != expected_size:
            raise ValueError(
                f"Sample ID count mismatch for {split}: raw={expected_size}, "
                f"semantic={len(semantic_ids)}"
            )
        for index, (raw_id, semantic_id) in enumerate(
            zip(raw_ids, semantic_ids)
        ):
            if raw_id != semantic_id:
                raise ValueError(
                    f"Semantic sample order mismatch at {split}[{index}]: "
                    f"raw ID={raw_id!r}, semantic ID={semantic_id!r}. "
                    'Regenerate semantic features from the same processed pickle.'
                )
        for modality in ('text', 'audio', 'vision'):
            token_key = f'semantic_{modality}'
            mask_key = f'semantic_{modality}_mask'
            if modality not in split_data or f'{modality}_mask' not in split_data:
                raise KeyError(
                    f"{semantic_path} must contain '{split}.{modality}' and "
                    f"'{split}.{modality}_mask'"
                )
            tokens = torch.as_tensor(split_data[modality])
            valid_mask = torch.as_tensor(split_data[f'{modality}_mask']).bool()
            if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"Invalid {split}.{modality} shapes: tokens={tuple(tokens.shape)}, "
                    f"mask={tuple(valid_mask.shape)}"
                )
            if tokens.size(0) != expected_size:
                raise ValueError(
                    f"Sample count mismatch for {split}.{modality}: "
                    f"raw={expected_size}, semantic={tokens.size(0)}"
                )
            self.data[token_key] = tokens
            self.data[mask_key] = valid_mask

    def __gen_mask(self):
        self.data['vision_padding_mask'] = lengths_to_padding_mask(
            self.data['vision_lengths'], self.data['vision'], 'vision'
        )
        self.data['audio_padding_mask'] = lengths_to_padding_mask(
            self.data['audio_lengths'], self.data['audio'], 'audio'
        )
    
    
    def __scale(self):
        self.scaled_audio = self.data['audio'].copy()
        self.scaled_vision = self.data['vision'].copy()
        self.scaled_text = self.data['text'].copy()
        for i in range(self.audio_fea_size[-1]):
            max_num = np.max(self.data['audio'][:, :, i])
            min_num = np.min(self.data['audio'][:, :, i])
            self.scaled_audio[:, :, i] = (self.data['audio'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        for i in range(self.vision_fea_size[-1]):
            max_num = np.max(self.data['vision'][:, :, i])
            min_num = np.min(self.data['vision'][:, :, i])
            self.scaled_vision[:, :, i] = (self.data['vision'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        self.scaled_audio = torch.tensor(self.scaled_audio)
        self.scaled_vision = torch.tensor(self.scaled_vision)
        self.scaled_text = torch.tensor(self.scaled_text)

    def __gen_cos_matrix(self, model=None):
        self.cos_matrix_M = torch.zeros((self.size, self.size))

        self.text_fea = torch.zeros((self.size, self.size))

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        if not self.scaled_embedding_averaged:
            # calculated mean
            audio_mean = torch.sum(self.scaled_audio, dim=-2)
            vision_mean = torch.sum(self.scaled_vision, dim=-2)
            text_mean = torch.sum(self.scaled_text, dim=-2)

            for i in range(len(audio_mean)):
                audio_mean[i, :] /= self.data['audio_lengths'][i]
            for i in range(len(vision_mean)):
                vision_mean[i, :] /= self.data['vision_lengths'][i]
            for i in range(len(text_mean)):
                text_mean[i, :] /= 50
        else:
            text_mean, vision_mean, audio_mean = self.scaled_text, self.scaled_vision, self.scaled_audio

        self.cos_matrix_M = cos(torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(1),
                                torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(0))

        self.rank_M = torch.zeros(self.cos_matrix_M.shape)

        for i in range(len(self.cos_matrix_M)):
            _, self.rank_M[i, :] = torch.sort(self.cos_matrix_M[i, :], descending=True)
        self.M_retrieve = self.__pre_sample(self.rank_M, np.round(self.data['regression_labels']))
        #self.M_retrieve = self.__pre_sample(self.rank_M, self.data['regression_labels'])

    def __pre_sample(self, _rank, _label):
        retrieve = {'ss': [],
                    'sd': [],
                    'ds': [],
                    'dd': [],
                    }
        for i in range(self.size):
            _ss = [] # feature same label same
            _sd = [] # feature same, label different
            _ds = [] # feature different, label same
            _dd = [] # feature different, label same
            for j in range(int(self.size)):
                if i == j: continue
                #if abs(_label[i] - _label[int(_rank[i][j])]) <= 0.2:
                if _label[i] == _label[int(_rank[i][j])]:
                    _ss.append(j)
                else:
                    _sd.append(j)
            for j in range(-1, -int(self.size), -1):
                if i == j: continue
                #if abs(_label[i] - _label[int(_rank[i][j])]) <= 0.2:
                if _label[i] == _label[int(_rank[i][j])]:
                    _ds.append(j)
                else:
                    _dd.append(j)
            if len(_ss) < 2 or len(_sd) < 2 or len(_ds) < 2 :
                print('Unique sample detected, may cause error!')
                print(len(_ss), len(_sd), len(_dd))
                for i in _ss:
                    print(_label[i])

            retrieve['ss'].append(_ss[:10])
            retrieve['sd'].append(_sd[:10])
            retrieve['ds'].append(_ds[:10])
            retrieve['dd'].append(_dd[:10])
        return retrieve

    def update_matrix(self, T, V, A):
        self.scaled_text = T
        self.scaled_vision = V
        self.scaled_audio = A
        self.scaled_embedding_averaged = True
        self.__gen_cos_matrix()

    def sample(self, _sample_idx):
        if not self.simi_return_mono:
            samples2 = {}
            idx2 = []
            for i in _sample_idx:
                idx2 += random.sample(self.M_retrieve['ss'][i], 2)
                idx2 += random.sample(self.M_retrieve['dd'][i], 2)
                idx2 += random.sample(self.M_retrieve['sd'][i], 2)
            for key in self.data:
                if type(self.data[key]) == list:
                    continue
                else:
                    if type(self.data[key][0]) == np.str_:
                        samples2[key] = self.data[key][idx2].tolist()
                    else:
                        if type(self.data[key]) is not torch.Tensor:
                            samples2[key] = torch.tensor(self.data[key][idx2])
                        else:
                            samples2[key] = self.data[key][idx2]
            return samples2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.data:
            samples[key] = self.data[key][idx]
        return samples


def MOSIDataloader(name, batch_size=None, use_sampler=False, use_similarity=False, simi_return_mono=False, shuffle=True,
                   num_workers=0, prefetch_factor=None, config=default_config):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = MOSIDataset(
        name,
        use_similarity=use_similarity,
        simi_return_mono=simi_return_mono,
        config=config,
    )
    sampler = None
    drop_last = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)
