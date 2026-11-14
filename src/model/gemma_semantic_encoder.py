import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class GemmaSemanticEncoder(nn.Module):
    """
    Offline-only token encoder for large-model generated descriptions.
    Input:
        list[str] descriptions
    Output:
        token features [B,L,H] and a valid-token mask [B,L]
    """
    def __init__(
        self,
        model_path,
        device=None,
        load_in_4bit=False,
        max_gpu_memory=None,
        max_cpu_memory='120GiB',
        offload_folder=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        if self.device.type == 'cuda':
            major, _ = torch.cuda.get_device_capability(self.device)
            model_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            model_dtype = torch.float32

        load_kwargs = {
            'local_files_only': True,
            'dtype': model_dtype,
        }
        use_device_map = load_in_4bit or max_gpu_memory is not None
        if use_device_map:
            load_kwargs['device_map'] = 'auto'
            load_kwargs['offload_state_dict'] = True
            if offload_folder:
                os.makedirs(offload_folder, exist_ok=True)
                load_kwargs['offload_folder'] = offload_folder
            if max_gpu_memory:
                load_kwargs['max_memory'] = {
                    **{
                        index: max_gpu_memory
                        for index in range(torch.cuda.device_count())
                    },
                    'cpu': max_cpu_memory,
                }
        if load_in_4bit:
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_quant_type='nf4',
            )

        self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
        if not use_device_map:
            self.model.to(self.device)
        else:
            self.device = self.model.device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad=False

    @torch.no_grad()
    def forward(self, texts, max_length=128):
        inputs=self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        # Keep an independent copy before the model forward. Some multimodal
        # model implementations transform the supplied attention mask in-place.
        valid_mask = inputs["attention_mask"].bool().clone()
        hidden=self.model(**inputs).last_hidden_state
        valid_mask = valid_mask.to(hidden.device)
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            indices = empty_rows.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f'Tokenizer returned zero valid tokens for batch rows {indices}'
            )
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return hidden, valid_mask
