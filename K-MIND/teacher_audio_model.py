"""K-MIND model zoo (1280-dim variant): text teacher, audio/video students, and ASF fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer, RobertaModel, TimesformerModel, Data2VecAudioModel
from transformers import AutoModel, AutoTokenizer

from transformers import WhisperModel

class Student_Audio_Whisper(nn.Module):
    def __init__(self, audio_model, clsNum, init_config=None):
        super().__init__()

        whisper = WhisperModel.from_pretrained(audio_model)
        self.encoder = whisper.encoder
        self.hiddenDim = whisper.config.d_model

        self.dropout = nn.Dropout(0.1)
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):
        # batch_input: [B, 128, T] for whisper-large-v3
        batch_input = batch_input.float()

        outputs = self.encoder(input_features=batch_input)
        hidden_states = outputs.last_hidden_state      # [B, L, H]

        # Whisper encoder에는 CLS 토큰 개념이 없어서 mean pooling이 무난
        batch_audio_output = hidden_states.mean(dim=1) # [B, H]
        batch_audio_output = self.dropout(batch_audio_output)

        audio_logit = self.W(batch_audio_output)       # [B, clsNum]
        return batch_audio_output, audio_logit

class Teacher_Text_Auto(nn.Module):
    def __init__(self, model_name, cls_num, vocab_size=None, kd_hidden_size=1280, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if vocab_size is not None:
            self.encoder.resize_token_embeddings(vocab_size)

        hidden_size = self.encoder.config.hidden_size  # roberta-large = 1024
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, cls_num)

        if hidden_size != kd_hidden_size:
            self.kd_proj = nn.Linear(hidden_size, kd_hidden_size)
        else:
            self.kd_proj = nn.Identity()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0]   # [B, 1024]

        logits = self.classifier(self.dropout(pooled))
        kd_hidden = self.kd_proj(self.dropout(pooled))   # [B, 1280]

        return kd_hidden, logits

class Student_Video_Auto(nn.Module):
    def __init__(self, model_name, cls_num, kd_hidden_size=1280, dropout=0.1):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        if hasattr(self.backbone.config, "hidden_size"):
            hidden_size = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"):
            hidden_size = self.backbone.config.d_model
        else:
            raise ValueError(
                f"Cannot infer hidden size from config of {model_name}"
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, cls_num)

        if hidden_size != kd_hidden_size:
            self.kd_proj = nn.Linear(hidden_size, kd_hidden_size)
        else:
            self.kd_proj = nn.Identity()

    def forward(self, pixel_values):
        outputs = self.backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            pooled = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError("Model output does not contain pooler_output or last_hidden_state")

        logits = self.classifier(self.dropout(pooled))
        kd_hidden = self.kd_proj(self.dropout(pooled))

        return kd_hidden, logits
                    
class Teacher_model(nn.Module):
    def __init__(self, text_model, clsNum):
        super(Teacher_model, self).__init__()
        
        """Text Model"""
        tmodel_path = text_model
        if text_model == 'klue/roberta-base':
            self.text_model = AutoModel.from_pretrained(tmodel_path)
            tokenizer = AutoTokenizer.from_pretrained(tmodel_path)
            self.speaker_list = ['<s1>', '<s2>'] #, '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
            self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
            tokenizer.add_special_tokens(self.speaker_tokens_dict)
            
        self.text_model.resize_token_embeddings(len(tokenizer))
        self.text_hiddenDim = self.text_model.config.hidden_size
        
        """Logit"""
        self.W = nn.Linear(self.text_hiddenDim, 1280)
        self.classifier = nn.Linear(1280, clsNum)

    def forward(self, batch_input_tokens, attention_masks):

        batch_context_output = self.text_model(batch_input_tokens, attention_masks).last_hidden_state[:,-1,:]
        
        batch_last_hidden = self.W(batch_context_output)
        context_logit = self.classifier(batch_last_hidden)

        return batch_last_hidden, context_logit


class Student_Audio(nn.Module):
    def __init__(self, audio_model, clsNum, init_config):
        super(Student_Audio, self).__init__()
        
        """Model Setting"""
        amodel_path = audio_model
        if audio_model == "facebook/data2vec-audio-base-960h":
            
            self.model = Data2VecAudioModel.from_pretrained(amodel_path)
            self.model.config.update(init_config.__dict__)

        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):

        batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 1280)
        audio_logit = self.W(batch_audio_output) # (batch, clsNum)
        
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
    """Self-attention over two modality tokens, independently for each sample."""
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

        self.TEXT_DIM = 1280
        self.VISUAL_DIM = 1280
        self.ACOUSTIC_DIM = 1280
        self.pairwise_attn = PairwiseModalityAttention(self.VISUAL_DIM, num_head)

        self.W_hav = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.ACOUSTIC_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        
        """Logit"""
        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        new_nv = self.pairwise_attn(visual, acoustic)
        
        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

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
        self.TEXT_DIM = 1280
        self.VISUAL_DIM = 1280
        self.VISUAL_DIM_2 = 1280
        self.pairwise_attn = PairwiseModalityAttention(self.VISUAL_DIM, num_head)

        self.W_hav = nn.Linear(self.VISUAL_DIM + self.VISUAL_DIM_2 + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.VISUAL_DIM_2, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.VISUAL_DIM_2)
        self.dropout = nn.Dropout(dropout_prob)
        
        """Logit"""
        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, visual, visual_2):
        eps = 1e-6
        new_nv = self.pairwise_attn(visual, visual_2)
        
        v_v_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((v_v_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(v_v_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

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
        self.mask_value = 50265
        # 0122
        self.dim = 1280
        self.avg_dist = []

        self.eps = 1e-8
        self.device = "cuda" if self.args.cuda else "cpu"
                
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )

        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        ).to(self.device)

        # self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo.pt").to(self.device)
        # self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        # 0121
        self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo_eng.pt").to(self.device)
        self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        
        anchor_input_dim = self.emo_anchor.shape[-1]
        self.map_function_anchor = nn.Sequential(
            nn.Linear(anchor_input_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        ).to(self.device)

        self.model = fusion_model

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def _forward(self, input_feature):
        mask_outputs = None
        mask_mapped_outputs = self.map_function(input_feature)
        feature = torch.dropout(input_feature, self.dropout, train=self.training)
        feature = self.predictor(feature)

        if self.args.use_nearest_neighbour:
            # anchors = self.map_function_anchor(self.emo_anchor)
            anchors = self.map_function_anchor(self.emo_anchor.to(input_feature.device))
            self.last_emo_anchor = anchors
            anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), anchors.unsqueeze(0))
            
        else:
            anchor_scores = None
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    
    def forward(self, feature_1280, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(feature_1280)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        else:
            return feature




# class CLModel(nn.Module):
#     def __init__(self, args, n_classes, fusion_model=None):
#         super().__init__()
#         self.args = args
#         self.dropout = args.dropout
#         self.num_classes = n_classes
#         self.mask_value = 50265
#         # 0122
#         self.dim = 1280
#         self.avg_dist = []

#         self.eps = 1e-8
#         self.device = "cuda" if self.args.cuda else "cpu"
                
#         self.predictor = nn.Sequential(
#             nn.Linear(self.dim, self.num_classes)
#         )

#         self.map_function_anchor = nn.Sequential(
#             # nn.Linear(1024, self.dim),
#             nn.Linear(self.dim, self.dim),
#             nn.LayerNorm(self.dim),
#             nn.ReLU(),
#             nn.Linear(self.dim, self.dim),
#         ).to(self.device)

#         self.map_function = nn.Sequential(
#             nn.Linear(self.dim, self.dim),
#             nn.LayerNorm(self.dim),
#             nn.ReLU(),
#             nn.Linear(self.dim, self.dim),
#         ).to(self.device)

#         # self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo.pt").to(self.device)
#         # self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
#         # 0121
#         self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo_eng.pt").to(self.device)
#         self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])

#         self.model = fusion_model

#     def score_func(self, x, y):
#         return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
#     def _forward(self, input_feature):
#         mask_outputs = None
#         mask_mapped_outputs = self.map_function(input_feature)
#         feature = torch.dropout(input_feature, self.dropout, train=self.training)
#         feature = self.predictor(feature)

#         if self.args.use_nearest_neighbour:
#             anchors = self.map_function_anchor(self.emo_anchor)

#             self.last_emo_anchor = anchors
#             anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), anchors.unsqueeze(0))
            
#         else:
#             anchor_scores = None
#         return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    
#     def forward(self, feature_1280, return_mask_output=False):
#         '''
#         generate vector representations for each turn of conversation
#         '''
#         feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(feature_1280)
        
#         if return_mask_output:
#             return feature, mask_mapped_outputs, mask_outputs, anchor_scores
#         else:
#             return feature


class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args
        
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
    
    def forward(self, emb):
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp

############################################################################################
class Student_Audio_PCM(nn.Module):
    def __init__(self,
                 audio_model_name: str,
                 clsNum: int,
                 embedding: str = "none",   # "none" | "linear" | "cnn"
                 attn_num_heads: int = 8,
                 pitch_hidden_size: int = 256,
                 do_pitch_norm: bool = False):
        super().__init__()
        self.embedding = embedding
        self.do_pitch_norm = do_pitch_norm

        self.model = Data2VecAudioModel.from_pretrained(audio_model_name)
        self.hiddenDim = self.model.config.hidden_size  # teacher가 1280이므로 보통 1280

        # classifier (emotion cls)
        self.W = nn.Linear(self.hiddenDim, clsNum)

        if self.embedding != "none":
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.hiddenDim,
                num_heads=attn_num_heads,
                batch_first=True,
            )

        if self.embedding == "linear":
            self.pitch_embedding = nn.Sequential(
                nn.Linear(1, pitch_hidden_size),
                nn.ReLU(),
                nn.Linear(pitch_hidden_size, self.hiddenDim),
            )
        elif self.embedding == "cnn":
            self.pitch_embedding = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, self.hiddenDim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, input_values, pitch=None, attention_mask=None):
        # input_values: [B, T]
        wav_hs = self.model(input_values).last_hidden_state  # [B, L_wav, H]

        if self.embedding == "none":
            pooled = wav_hs.mean(dim=1)  # [B, H]
        else:
            assert pitch is not None, "embedding != none이면 pitch가 필요합니다."

            if self.do_pitch_norm:
                mean = pitch.mean(dim=1, keepdim=True)
                std = pitch.std(dim=1, keepdim=True)
                pitch = (pitch - mean) / (std + 1e-8)

            if self.embedding == "linear":
                pitch_emb = self.pitch_embedding(pitch.unsqueeze(-1))  # [B, Lp, H]
            else:  # cnn
                x = self.pitch_embedding(pitch.unsqueeze(1))          # [B, H, Lp]
                pitch_emb = x.transpose(1, 2)                         # [B, Lp, H]

            attn_out, _ = self.attention_layer(
                query=pitch_emb,
                key=wav_hs,
                value=wav_hs,
            )  # [B, Lp, H]
            pooled = attn_out.mean(dim=1)  # [B, H]

        logits = self.W(pooled)  # [B, clsNum]
        return pooled, logits


class ASF_PCM(nn.Module):
    def __init__(self, clsNum, hidden_size=1280, beta_shift=1.0, dropout_prob=0.1, num_head=8):
        super().__init__()

        self.TEXT_DIM = 1280
        self.VISUAL_DIM = 1280
        self.ACOUSTIC_DIM = 1280
        self.PCM_DIM = 1280

        self.NV_DIM = self.VISUAL_DIM + self.ACOUSTIC_DIM + self.PCM_DIM  # 2304

        # MultiheadAttention: embed_dim == NV_DIM이어야 함
        self.multihead_attn = nn.MultiheadAttention(self.NV_DIM, num_head, batch_first=True)

        # gating
        self.W_hav = nn.Linear(self.NV_DIM + self.TEXT_DIM, self.TEXT_DIM)  # [nv + text] -> 1280
        self.W_av  = nn.Linear(self.NV_DIM, self.TEXT_DIM)                  # nv -> 1280

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)          # 1280
        self.NV_LayerNorm = nn.LayerNorm(self.NV_DIM)       # 2304
        self.dropout = nn.Dropout(dropout_prob)

        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, visual, acoustic, pitch):
        """
        text_embedding: [B, 1280]
        visual:        [B, 1280]
        acoustic:      [B, 1280]
        pitch:         [B, 1280]
        """
        eps = 1e-6

        # nv concat: [B, 2304]
        nv = torch.cat((visual, acoustic, pitch), dim=-1)

        # attention은 3D를 기대하니까 [B, 1, 2304]로 바꿔줌
        nv_seq = nv.unsqueeze(1)  # [B, 1, 2304]

        # self-attn
        new_nv = self.multihead_attn(nv_seq, nv_seq, nv_seq)[0] + nv_seq  # [B, 1, 2304]
        new_nv = new_nv.squeeze(1)  # [B, 2304]

        nv_embedd = self.dropout(self.NV_LayerNorm(new_nv))  # [B, 2304]

        weight_av = F.relu(self.W_hav(torch.cat((nv_embedd, text_embedding), dim=-1)))  # [B,1280]
        h_m = weight_av * self.W_av(nv_embedd)  # [B,1280]

        em_norm = text_embedding.norm(2, dim=-1)  # [B]
        hm_norm = h_m.norm(2, dim=-1)             # [B]
        hm_norm = torch.where(hm_norm == 0, torch.ones_like(hm_norm), hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        alpha = torch.min(thresh_hold, torch.ones_like(thresh_hold)).unsqueeze(-1)  # [B,1]

        fused = alpha * h_m  # [B,1280]

        embedding_output = self.dropout(self.LayerNorm(fused + text_embedding))  # [B,1280]
        logits = self.W(embedding_output)  # [B,clsNum]

        return embedding_output, logits
