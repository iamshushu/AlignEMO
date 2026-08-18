"""ELACL hybrid loss for K-MIND: CE + supervised contrastive loss with emotion anchors (Ekman 7-way)."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class HybridLossOutput:
    ce_loss:torch.Tensor = None
    cl_loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None

def loss_function(log_prob, reps, label, mask, model):
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(reps.device)
    scl_loss_fn = SupConLoss(model.args)
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=cl_loss.loss,
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
    ) 

def AngleLoss(means):
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss, max_cosine

@dataclass
class SupConOutput:
    loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None


class SupConLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temp
        self.eps = 1e-8
        if args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        else:
            # self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo.pt")
            # self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            # self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo_ekman.pt")
            self.emo_anchor = torch.load(f"{args.anchor_path}/kmind_emo_eng.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        self.sim = nn.functional.cosine_similarity(self.emo_anchor.unsqueeze(1), self.emo_anchor.unsqueeze(0), dim=2)
        self.args = args

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def forward(self, reps, labels, model, return_representations=False):
        device = reps.device
        batch_size = reps.shape[0]
        self.emo_anchor = self.emo_anchor.to(device)
        self.emo_label = self.emo_label.to(device)
        emo_anchor = model.map_function_anchor(self.emo_anchor)
        if return_representations:
            sentiment_labels = labels
            sentiment_representations = reps.detach()
            sentiment_anchortypes = emo_anchor.detach()
        else:
            sentiment_labels = None
            sentiment_representations = None
            sentiment_anchortypes = None
        if self.args.disable_emo_anchor:
            concated_reps = reps
            concated_labels = labels
            concated_bsz = batch_size
        else:
            concated_reps = torch.cat([reps, emo_anchor], dim=0)
            concated_labels = torch.cat([labels, self.emo_label], dim=0)
            concated_bsz = batch_size + emo_anchor.shape[0]
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)
        
        scores /= self.temperature
        scores = scores[:concated_bsz]
        pos_mask = pos_mask[:concated_bsz]
        mask = mask[:concated_bsz]
        
        scores -= torch.max(scores).item()

        angleloss, max_cosine = AngleLoss(emo_anchor)
        # print(max_cosine)

        scores = torch.exp(scores)
        if pos_mask.shape[1] > scores.shape[1]:
            pos_mask = pos_mask[:, :scores.shape[1]]
        elif pos_mask.shape[1] < scores.shape[1]:
            pos_mask = F.pad(pos_mask, (0, scores.shape[1] - pos_mask.shape[1]))

        pos_scores = scores * (pos_mask * mask)
        neg_scores = scores * (1 - pos_mask)
        probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
        positive_counts = (pos_mask * mask).sum(-1)
        probs /= positive_counts + self.eps
        loss = - torch.log(probs + self.eps)
        query_mode = getattr(self.args, "supcon_query_mode", "current")
        if query_mode == "current":
            query_mask = loss > 0.0
        elif query_mode == "valid_query":
            query_mask = positive_counts > 0
        elif query_mode == "utterance_only":
            query_mask = torch.arange(concated_bsz, device=device) < batch_size
        else:
            raise ValueError(f"unknown supcon_query_mode: {query_mode}")
        loss = loss[query_mask].mean()

        loss += self.args.angle_loss_weight * angleloss
        return SupConOutput(
            loss=loss,
            sentiment_representations=sentiment_representations,
            sentiment_labels=sentiment_labels,
            sentiment_anchortypes=sentiment_anchortypes,
            anchortype_labels=self.emo_label,
            max_cosine = max_cosine
        )
    
