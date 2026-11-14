"""SERA composition: original encoders, semantic selection, explanation alignment and text-guided fusion. Keep attribute names stable for checkpoints."""

import torch
from torch import nn
import os
import hashlib

from model.text_encoder import TextEncoder
from model.vision_encoder import VisionEncoder
from model.audio_encoder import AudioEncoder
from model.layers import TemporalAlignment
from model.MLP import BaseClassifier
from model.multimodal_fast_pathway import MultimodalFastPathway
from model.semantic_explanation_alignment import SemanticExplanationAlignment
from model.text_semantic_self_attention import TextSemanticSelfAttention
from model.late_fusion import LateFusionAdapter
import config as default_config
from utils import check_dir

class SERA(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, d_model = None, depth = None, drop_out=None, config=default_config):
        super(SERA, self).__init__()
        self.config = config
        if encoder_fea_dim is None:
            encoder_fea_dim = config.PARAM.downStream.encoder_fea_dim
        if d_model is None:
            d_model = config.PARAM.downStream.d_model
        if drop_out is None:
            drop_out = config.PARAM.downStream.drop_out
        if depth is None:
            depth = config.PARAM.downStream.fusion_depth

        # feature extractor
        self.text_encoder = TextEncoder(config=config)
        self.vision_encoder = VisionEncoder(config=config)
        self.audio_encoder = AudioEncoder(config=config)

        self.use_gemma_semantic = config.PARAM.downStream.use_gemma_semantic
        if self.use_gemma_semantic:
            self.use_semantic_selection = getattr(
                config.PARAM.downStream,
                'semantic_selection',
                True,
            )
            self.use_semantic_auxiliary = getattr(
                config.PARAM.downStream,
                'semantic_auxiliary',
                True,
            )
            # The input dimension is materialized from the offline feature file
            # on the first forward pass. This supports different Gemma variants.
            self.semantic_projectors = nn.ModuleDict({
                modality: nn.Sequential(
                    nn.LazyLinear(encoder_fea_dim),
                    nn.LayerNorm(encoder_fea_dim),
                )
                for modality in ('text', 'audio', 'vision')
            })
            if self.use_semantic_selection:
                semantic_top_k = {
                    'text': getattr(
                        config.PARAM.downStream,
                        'semantic_text_top_k',
                        config.PARAM.downStream.semantic_top_k,
                    ),
                    'audio': getattr(
                        config.PARAM.downStream,
                        'semantic_audio_top_k',
                        config.PARAM.downStream.semantic_top_k,
                    ),
                    'vision': getattr(
                        config.PARAM.downStream,
                        'semantic_vision_top_k',
                        config.PARAM.downStream.semantic_top_k,
                    ),
                }
                self.semantic_fast = nn.ModuleDict({
                    modality: MultimodalFastPathway(
                        encoder_fea_dim,
                        semantic_top_k[modality],
                    )
                    for modality in ('text', 'audio', 'vision')
                })
                if self.use_semantic_auxiliary:
                    self.semantic_aux_classifiers = nn.ModuleDict({
                        modality: nn.Sequential(
                            nn.Linear(encoder_fea_dim, encoder_fea_dim),
                            nn.ReLU(),
                            nn.Linear(encoder_fea_dim, 1),
                        )
                        for modality in ('text', 'audio', 'vision')
                    })
            self.use_semantic_explanation_alignment = getattr(
                config.PARAM.downStream,
                'semantic_explanation_alignment',
                True,
            )
            if self.use_semantic_explanation_alignment:
                self.semantic_explanation_aligners = nn.ModuleDict({
                    modality: SemanticExplanationAlignment(
                        encoder_fea_dim,
                        num_heads=8,
                        dropout=drop_out,
                        gated=getattr(
                            config.PARAM.downStream,
                            'semantic_alignment_gated',
                            False,
                        ),
                        gate_init=getattr(
                            config.PARAM.downStream,
                            'semantic_alignment_gate_init',
                            0.1,
                        ),
                    )
                    # TEXT aligns generated explanations to low-level audio and
                    # visual features.  Its text stream remains the fusion router.
                    for modality in ('audio', 'vision')
                })

        self.fusion_pool_score = nn.Sequential(
            nn.Linear(encoder_fea_dim, encoder_fea_dim),
            nn.Tanh(),
            nn.Linear(encoder_fea_dim, 1),
        )

        self.use_text_guided_fusion = getattr(
            config.PARAM.downStream,
            'text_guided_fusion',
            True,
        )
        self.use_text_guided_modality_gate = getattr(
            config.PARAM.downStream,
            'text_guided_modality_gate',
            False,
        )
        self.no_tgf_fusion_mode = getattr(
            config.PARAM.downStream,
            'no_tgf_fusion_mode',
            'legacy_concat',
        )
        self.use_text_semantic_self_attention = (
            self.use_gemma_semantic
            and getattr(
                config.PARAM.downStream,
                'text_semantic_self_attention',
                False,
            )
        )
        if self.use_text_semantic_self_attention:
            self.text_semantic_self_attention = TextSemanticSelfAttention(
                encoder_fea_dim,
                num_heads=getattr(
                    config.PARAM.downStream,
                    'text_semantic_self_attention_heads',
                    8,
                ),
                num_layers=getattr(
                    config.PARAM.downStream,
                    'text_semantic_self_attention_layers',
                    1,
                ),
                dropout=drop_out,
            )
        if self.use_text_guided_fusion:
            # Keep one explicit text-Q / audio-video-KV cross-attention block.
            self.temporal_alignment = TemporalAlignment(
                encoder_fea_dim,
                8,
                32,
                drop_out,
                modality_gate=self.use_text_guided_modality_gate,
            )

        if not self.use_text_guided_fusion:
            if self.no_tgf_fusion_mode == 'adapter':
                self.no_tgf_adapter = LateFusionAdapter(encoder_fea_dim)
            elif self.no_tgf_fusion_mode not in {'legacy_concat', 'mean'}:
                raise ValueError(
                    'no_tgf_fusion_mode must be legacy_concat, adapter or mean'
                )

        hidden_size = [d_model * 2, d_model]
        classifier_input_size = encoder_fea_dim
        if (
            not self.use_text_guided_fusion
            and self.no_tgf_fusion_mode == 'legacy_concat'
        ):
            classifier_input_size = encoder_fea_dim * 3
        self.classifier = BaseClassifier(
            input_size=classifier_input_size,
            hidden_size=hidden_size,
            output_size=1,
        )

        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.set_train()

    def _select_semantic_tokens(self, sample, modality, device,
                                return_explanation=False):
        token_key = f'semantic_{modality}'
        mask_key = f'semantic_{modality}_mask'
        if token_key not in sample or mask_key not in sample:
            raise KeyError(
                f"Semantic enhancement is enabled but '{token_key}'/'{mask_key}' "
                "is absent. Generate and load offline semantic token features first."
            )
        tokens = sample[token_key].to(device=device, dtype=torch.float32)
        valid_mask = sample[mask_key].to(device=device, dtype=torch.bool)
        tokens = self.semantic_projectors[modality](tokens)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        if self.use_semantic_selection:
            selected, scores, selected_valid, pooled, indices = self.semantic_fast[modality](
                tokens, valid_mask
            )
            if self.use_semantic_auxiliary:
                auxiliary_pred = self.semantic_aux_classifiers[modality](
                    pooled
                ).squeeze(-1)
            else:
                auxiliary_pred = None
        else:
            # No-selector ablation: preserve every valid semantic token in its
            # original order. There is no score network, hard Top-K operation,
            # straight-through gate, or selector auxiliary regression head.
            selected = tokens
            selected_valid = valid_mask
            auxiliary_pred = None
            indices = torch.arange(
                tokens.size(1), device=tokens.device
            ).unsqueeze(0).expand(tokens.size(0), -1)
            scores = valid_mask.to(tokens.dtype)
        if return_explanation:
            selected_scores = torch.gather(scores, 1, indices)
            explanation = {
                'indices': indices.detach(),
                'scores': scores.detach(),
                'selected_scores': selected_scores.detach(),
                'selected_valid': selected_valid.detach(),
                'valid_mask': valid_mask.detach(),
            }
            return selected, selected_valid, auxiliary_pred, explanation
        return selected, selected_valid, auxiliary_pred

    def _pool_sequence(self, sequence, padding_mask):
        """Attention-pool one modality while excluding padded positions."""
        scores = self.fusion_pool_score(sequence).squeeze(-1)
        scores = scores.masked_fill(
            padding_mask.bool(), torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        valid = (~padding_mask.bool()).to(sequence.dtype)
        weights = weights * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)

    def _pool_text_queries(self, aligned_text, text_padding_mask):
        return self._pool_sequence(aligned_text, text_padding_mask)

    def forward(self, sample1, sample2=None, return_loss=True, device=None,
                aux_weight=1.0, return_explanations=False):
        if device is None:
            device = self.device

        text1 = sample1['raw_text']
        vision1 = sample1['vision'].clone().detach().to(device).float()
        audio1 = sample1['audio'].clone().detach().to(device).float()
        label1 = sample1['regression_labels'].clone().detach().to(device).float().view(-1)
        key_padding_mask_V1, key_padding_mask_A1 = (sample1['vision_padding_mask'].clone().detach().to(device),
                                                    sample1['audio_padding_mask'].clone().detach().to(device))
        
        t_embed_all, t_embed, key_padding_mask_T1 = self.text_encoder(
            text1, device=device, return_padding_mask=True
        )
        a_embed_all, a_embed = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1, device=device) #.squeeze()
        v_embed_all, v_embed = self.vision_encoder(
            vision1,
            key_padding_mask=key_padding_mask_V1,
            device=device
        )

        # Select important tokens from each offline semantic stream, then append
        # them to that modality's original encoded sequence.
        auxiliary_predictions = []
        semantic_explanations = {}
        if self.use_gemma_semantic:
            semantic_selections = {}
            for modality in ('text', 'audio', 'vision'):
                selection = self._select_semantic_tokens(
                    sample1,
                    modality,
                    device,
                    return_explanation=return_explanations,
                )
                if return_explanations:
                    selected, selected_valid, auxiliary_pred, explanation = selection
                    semantic_explanations[modality] = explanation
                else:
                    selected, selected_valid, auxiliary_pred = selection
                semantic_selections[modality] = (
                    selected, selected_valid, auxiliary_pred
                )

            gt, gt_valid, gt_pred = semantic_selections['text']
            ga, ga_valid, ga_pred = semantic_selections['audio']
            gv, gv_valid, gv_pred = semantic_selections['vision']
            auxiliary_predictions = [
                prediction
                for prediction in (gt_pred, ga_pred, gv_pred)
                if prediction is not None
            ]

            # TEXT explanation alignment: audio/video explanations are queries;
            # their original low-level streams supply keys and values.  The text
            # Summary is already a textual semantic enhancement and is appended
            # directly, matching the role of text as the later fusion router.
            if self.use_semantic_explanation_alignment:
                ga = self.semantic_explanation_aligners['audio'](
                    ga,
                    a_embed_all,
                    explanation_valid=ga_valid,
                    feature_padding_mask=key_padding_mask_A1,
                )
                gv = self.semantic_explanation_aligners['vision'](
                    gv,
                    v_embed_all,
                    explanation_valid=gv_valid,
                    feature_padding_mask=key_padding_mask_V1,
                )

            t_embed_all = torch.cat((t_embed_all, gt), dim=1)
            a_embed_all = torch.cat((a_embed_all, ga), dim=1)
            v_embed_all = torch.cat((v_embed_all, gv), dim=1)
            key_padding_mask_T1 = torch.cat((key_padding_mask_T1, ~gt_valid), dim=1)
            key_padding_mask_A1 = torch.cat((key_padding_mask_A1, ~ga_valid), dim=1)
            key_padding_mask_V1 = torch.cat((key_padding_mask_V1, ~gv_valid), dim=1)

            # Optional joint contextualization of original BERT tokens and
            # selected text-semantic anchors before text-guided fusion.
            if self.use_text_semantic_self_attention:
                t_embed_all = self.text_semantic_self_attention(
                    t_embed_all,
                    padding_mask=key_padding_mask_T1,
                )

        # Keep the original no-loss interface unchanged.
        if not return_loss:
            if return_explanations:
                return (t_embed, v_embed, a_embed), semantic_explanations
            return (t_embed, v_embed, a_embed)

        if self.use_text_guided_fusion:
            # Text (original + selected semantic tokens) queries audio and
            # vision, producing the full model's text-centred representation.
            t_a_v_embed_all = self.temporal_alignment(
                t_embed_all,
                a_embed_all,
                v_embed_all,
                text_padding_mask=key_padding_mask_T1,
                audio_padding_mask=key_padding_mask_A1,
                vision_padding_mask=key_padding_mask_V1,
            )
            t_a_v_embed = self._pool_text_queries(
                t_a_v_embed_all, key_padding_mask_T1
            )
        else:
            # -TGF: no modality queries another modality.  The fair ``adapter``
            # mode compresses pooled T/A/V from 3D to D before the unchanged
            # D-dimensional prediction head. ``legacy_concat`` is retained only
            # for loading and reproducing old ablation checkpoints.
            pooled_text = self._pool_sequence(
                t_embed_all, key_padding_mask_T1
            )
            pooled_audio = self._pool_sequence(
                a_embed_all, key_padding_mask_A1
            )
            pooled_vision = self._pool_sequence(
                v_embed_all, key_padding_mask_V1
            )
            if self.no_tgf_fusion_mode == 'adapter':
                t_a_v_embed = self.no_tgf_adapter(
                    pooled_text, pooled_audio, pooled_vision
                )
            elif self.no_tgf_fusion_mode == 'mean':
                t_a_v_embed = (
                    pooled_text + pooled_audio + pooled_vision
                ) / 3.0
            else:
                t_a_v_embed = torch.cat(
                    (pooled_text, pooled_audio, pooled_vision), dim=-1
                )

        pred = self.classifier(t_a_v_embed).squeeze(-1)
        pred_loss = self.criterion(pred, label1)

        auxiliary_loss = pred_loss.new_zeros(())
        if auxiliary_predictions:
            modality_aux_losses = torch.stack([
                self.criterion(auxiliary_pred, label1)
                for auxiliary_pred in auxiliary_predictions
            ])
            reduction = getattr(
                self.config.PARAM.downStream,
                'semantic_aux_reduction',
                'mean',
            )
            if reduction == 'mean':
                auxiliary_loss = modality_aux_losses.mean()
            elif reduction == 'sum':
                auxiliary_loss = modality_aux_losses.sum()
            else:
                raise ValueError(
                    "semantic_aux_reduction must be either 'mean' or 'sum', "
                    f"got {reduction!r}"
                )
        loss = pred_loss + float(aux_weight) * auxiliary_loss

        if return_explanations:
            return pred, loss, pred_loss, auxiliary_loss, semantic_explanations
        return pred, loss, pred_loss, auxiliary_loss

    def save_model(self, name, dataset):
        # save all modules
        if dataset == 'mosi':
            model_path = self.config.MOSI.model_path + name + '_model.ckpt'
        elif dataset == 'mosei':
            model_path = self.config.MOSEI.model_path + name + '_model.ckpt'
        elif dataset == 'sims':
            model_path = self.config.SIMS.model_path + name + '_model.ckpt'
        check_dir(os.path.dirname(model_path))
        print('model saved at:')
        print(model_path)
        temporary_path = model_path + '.tmp'
        with open(temporary_path, 'wb') as file:
            torch.save(self.state_dict(), file)
            file.flush()
            os.fsync(file.fileno())
        # Atomic replacement preserves the previous complete checkpoint if the
        # process is interrupted while writing the temporary file.
        os.replace(temporary_path, model_path)

        checksum = self._sha256_file(model_path)
        checksum_path = model_path + '.sha256'
        checksum_temporary_path = checksum_path + '.tmp'
        with open(checksum_temporary_path, 'w', encoding='ascii') as file:
            file.write(f'{checksum}  {os.path.basename(model_path)}\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(checksum_temporary_path, checksum_path)
        print(f'SHA-256: {checksum}')

    def load_model(self, name, dataset):

        if dataset == 'mosi':
            model_path = self.config.MOSI.model_path + name + '_model.ckpt'
        elif dataset == 'mosei':
            model_path = self.config.MOSEI.model_path + name + '_model.ckpt'
        elif dataset == 'sims':
            model_path = self.config.SIMS.model_path + name + '_model.ckpt'
        print('model loaded from:')
        print(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Checkpoint does not exist: {model_path}')

        checksum_path = model_path + '.sha256'
        if os.path.isfile(checksum_path):
            with open(checksum_path, 'r', encoding='ascii') as file:
                expected_checksum = file.read().strip().split()[0].lower()
            actual_checksum = self._sha256_file(model_path)
            if actual_checksum != expected_checksum:
                raise RuntimeError(
                    f'Checkpoint checksum mismatch: {model_path}. Expected '
                    f'{expected_checksum}, got {actual_checksum}. The file may '
                    'be incomplete or corrupted; copy it again before loading.'
                )
        else:
            print(
                f'warning: no checksum sidecar found for legacy checkpoint: '
                f'{checksum_path}'
            )

        try:
            state_dict = torch.load(model_path, map_location=self.device)
        except Exception as error:
            raise RuntimeError(
                f'Failed to read checkpoint {model_path}. It is likely '
                'incomplete or corrupted. Re-copy it and compare SHA-256 '
                'checksums on the server and local machine.'
            ) from error
        self.load_state_dict(state_dict)

    @staticmethod
    def _sha256_file(path, chunk_size=8 * 1024 * 1024):
        digest = hashlib.sha256()
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True, True, True]

        for param in self.parameters():
            param.requires_grad = train_module[3]
        self.text_encoder.set_train(train_module=train_module[0:2])
        self.vision_encoder.set_train(train_module=train_module[2])
        self.audio_encoder.set_train(train_module=train_module[2])
