"""Model definitions for MELD: teacher (text), audio/video students, HMF fusion (ASF), and ELACL CLModel/Classifier."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
from transformers import (
    TimesformerModel, Data2VecAudioModel, WhisperModel,
    AutoTokenizer, AutoModel,
)

class Student_Audio_Whisper(nn.Module):
    def __init__(self, audio_model, clsNum, init_config=None):
        super(Student_Audio_Whisper, self).__init__()

        self.model = WhisperModel.from_pretrained(audio_model)
        self.hiddenDim = self.model.config.d_model

        # teacher / fusion 쪽 768에 맞추기 위한 projection
        self.proj = nn.Linear(self.hiddenDim, 768)
        self.W = nn.Linear(768, clsNum)

    def forward(self, batch_input_features):
        batch_audio_output = self.model.encoder(batch_input_features).last_hidden_state[:, 0, :]
        batch_audio_output = self.proj(batch_audio_output)
        audio_logit = self.W(batch_audio_output)

        return batch_audio_output, audio_logit
        
class Teacher_model(nn.Module):
    def __init__(self, text_model, clsNum):
        super(Teacher_model, self).__init__()

        """Text Model — AutoModel/AutoTokenizer supports roberta-large,
        microsoft/deberta-v3-large, and other HF encoders."""
        tmodel_path = text_model
        self.text_model = AutoModel.from_pretrained(tmodel_path)
        tokenizer = AutoTokenizer.from_pretrained(tmodel_path)
        self.speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
        self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
        tokenizer.add_special_tokens(self.speaker_tokens_dict)

        self.text_model.resize_token_embeddings(len(tokenizer))
        self.text_hiddenDim = self.text_model.config.hidden_size

        """Logit"""
        self.W = nn.Linear(self.text_hiddenDim, 768)
        self.classifier = nn.Linear(768, clsNum)

    def forward(self, batch_input_tokens, attention_masks):

        batch_context_output = self.text_model(batch_input_tokens, attention_masks).last_hidden_state[:,-1,:] # (batch, 1024)
        
        batch_last_hidden = self.W(batch_context_output)
        context_logit = self.classifier(batch_last_hidden)

        return batch_last_hidden, context_logit



class Student_Audio(nn.Module):
    def __init__(self, audio_model, clsNum, init_config, target_dim=768):
        super(Student_Audio, self).__init__()

        """Model Setting — Data2VecAudio kept for legacy path; other
        encoders (microsoft/wavlm-large, microsoft/hubert-large, ...)
        load through AutoModel."""
        amodel_path = audio_model
        if audio_model == "facebook/data2vec-audio-base-960h":
            self.model = Data2VecAudioModel.from_pretrained(amodel_path)
            self.model.config.update(init_config.__dict__)
        else:
            self.model = AutoModel.from_pretrained(amodel_path)
            if init_config is not None:
                for k, v in init_config.__dict__.items():
                    if hasattr(self.model.config, k):
                        setattr(self.model.config, k, v)

        self.hiddenDim = self.model.config.hidden_size
        self.target_dim = target_dim

        """Project to shared fusion dim (768) when encoder hidden != 768"""
        if self.hiddenDim != target_dim:
            self.proj = nn.Linear(self.hiddenDim, target_dim)
        else:
            self.proj = nn.Identity()

        """score"""
        self.W = nn.Linear(target_dim, clsNum)

    def forward(self, batch_input):

        batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:]
        batch_audio_output = self.proj(batch_audio_output)
        audio_logit = self.W(batch_audio_output)

        return batch_audio_output, audio_logit


class Student_Video(nn.Module):
    def __init__(self, video_model, clsNum):
        super(Student_Video, self).__init__()
        
        vmodel_path = video_model
        if video_model == "facebook/timesformer-base-finetuned-k400":

            self.model = TimesformerModel.from_pretrained(vmodel_path)

        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):

        batch_video_output = self.model(batch_input).last_hidden_state[:,0,:]
        video_logit = self.W(batch_video_output) 
        
        return batch_video_output, video_logit


class PairwiseModalityAttention(nn.Module):
    """Self-attention over exactly two modality tokens, independently per sample."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, first, second):
        if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
            raise ValueError(f"Expected matching [batch, hidden] tensors; got {tuple(first.shape)} and {tuple(second.shape)}")
        tokens = torch.stack((first, second), dim=1)
        attended, _ = self.multihead_attn(tokens, tokens, tokens, need_weights=False)
        return (attended + tokens).flatten(start_dim=1)


class ASF(nn.Module):
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super(ASF, self).__init__()

        self.TEXT_DIM = 768
        self.VISUAL_DIM = 768
        self.ACOUSTIC_DIM = 768
        self.pairwise_attn = PairwiseModalityAttention(self.VISUAL_DIM, num_head)

        self.W_hav = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.ACOUSTIC_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        
        """Logit"""
        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, *, visual, acoustic):
        eps = 1e-6
        new_nv = self.pairwise_attn(visual, acoustic)
        
        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones_like(hm_norm)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones_like(thresh_hold)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )
        
        logits = self.W(embedding_output)
        return embedding_output, logits  



class ASF_visual(nn.Module):
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super(ASF_visual, self).__init__()
        self.TEXT_DIM = 768
        self.VISUAL_DIM = 768
        self.VISUAL_DIM_2 = 768
        self.pairwise_attn = PairwiseModalityAttention(self.VISUAL_DIM, num_head)

        self.W_hav = nn.Linear(self.VISUAL_DIM + self.VISUAL_DIM_2 + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.VISUAL_DIM_2, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.VISUAL_DIM_2)
        self.dropout = nn.Dropout(dropout_prob)
        
        """Logit"""
        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, *, visual, visual_2):
        eps = 1e-6
        new_nv = self.pairwise_attn(visual, visual_2)
        
        v_v_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((v_v_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(v_v_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones_like(hm_norm)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones_like(thresh_hold)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        vis2_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(vis2_vis_embedding + text_embedding)
        )
        
        logits = self.W(embedding_output)
        return embedding_output, logits  




class CLModel(nn.Module):
    def __init__(self, args, n_classes, fusion_model=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = n_classes

        text_model = getattr(args, "text_model", "roberta-large")
        _tmp_tok = AutoTokenizer.from_pretrained(text_model)
        self.mask_value = _tmp_tok.mask_token_id if _tmp_tok.mask_token_id is not None else 50265

        self.dim = 768
        self.avg_dist = []

        self.eps = 1e-8
        self.device = "cuda" if self.args.cuda else "cpu"
                
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )

        self.map_function_anchor = nn.Sequential(
            nn.Linear(1024, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        ).to(self.device)

        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        ).to(self.device)

        self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt").to(self.device)
        self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        self.model = fusion_model

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def _forward(self, input_feature):
        mask_outputs = None
        mask_mapped_outputs = self.map_function(input_feature)
        feature = torch.dropout(input_feature, self.dropout, train=self.training)
        feature = self.predictor(feature)

        if self.args.use_nearest_neighbour:
            anchors = self.map_function_anchor(self.emo_anchor)

            self.last_emo_anchor = anchors
            anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), anchors.unsqueeze(0))
            
        else:
            anchor_scores = None
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    
    def forward(self, feature_768, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(feature_768)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        else:
            return feature


class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args
        
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
    
    def forward(self, emb):
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp
