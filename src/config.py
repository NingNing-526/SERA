"""SERA configuration adapter; runtime settings are supplied by run.py."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[1]
settings = json.loads(os.environ.get('SERA_SETTINGS', (ROOT / 'configs/default.json').read_text(encoding='utf-8')))
dataset = settings['dataset']
seed = int(settings['seed'])
USEROBERTA = False
DEVICE = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if settings['device'] == 'auto' else settings['device'])
bert_model_path = settings['bert_path']
variant = settings['variant']
depth = 4 if dataset == 'mosei' else 2
downstream = NS(
    metric='Loss', check_list=['Loss'], load_metric='best_Loss', check={'Loss': float('inf')},
    use_reg=True, encoder_fea_dim=128, text_fea_dim=768, d_model=128,
    experiment_variant=variant, use_gemma_semantic=variant != 'wo_sem', semantic_required=True,
    semantic_top_k=8, semantic_text_top_k=settings['top_k'][0],
    semantic_audio_top_k=settings['top_k'][1], semantic_vision_top_k=settings['top_k'][2],
    semantic_selection=variant != 'wo_ret', semantic_explanation_alignment=variant != 'wo_align',
    semantic_auxiliary=variant not in ('wo_aux', 'wo_ret'), semantic_aux_reduction='mean',
    semantic_aux_start_weight=settings['aux_start'], semantic_aux_min_weight=settings['aux_min'],
    semantic_alignment_gated=False, semantic_alignment_gate_init=0.1,
    text_guided_fusion=variant != 'wo_tgf', text_guided_modality_gate=False,
    no_tgf_fusion_mode='legacy_concat', text_semantic_self_attention=False,
    drop_out=settings['dropout'], t_drop_out=0.1, a_drop_out=0.1, v_drop_out=0.1,
    audio_nhead=8, vision_nhead=8, audio_tf_num_layers=depth, vision_tf_num_layers=depth,
    fusion_depth=3 if dataset == 'mosei' else 2, heat=0.5,
)
PARAM = NS(downStream=downstream, Train=NS(
    batch_size=settings['batch_size'], epoch=settings['epochs'], lr=settings['encoder_lr'],
    encoder_lr=settings['encoder_lr'], new_module_lr=settings['new_module_lr'],
    decay=settings['weight_decay'], num_warm_up=settings['warmup_epochs'], grad_clip_norm=1.0,
    early_stopping_patience=settings['early_stop_patience'],
    early_stopping_min_epochs=settings['early_stop_min_epochs'],
    early_stopping_min_delta=settings['early_stop_min_delta'], early_stopping_metric='Loss',
))
def data_config(audio_dim, vision_dim):
    out = Path(settings['output'])
    return NS(raw_data_path=settings['raw_data'], semantic_feature_path=settings['semantic_features'],
              audio_fea_dim=audio_dim, vision_fea_dim=vision_dim,
              model_path=str(out / 'checkpoints') + '/', log_path=str(out / 'logs') + '/',
              result_path=str(out / 'results') + '/')
MOSI = data_config(5, 20)
MOSEI = data_config(74, 35)
SIMS = data_config(33, 709)  # internal legacy import compatibility; not a supported CLI dataset
