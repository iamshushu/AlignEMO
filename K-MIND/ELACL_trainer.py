"""ELACL train/eval loops for K-MIND: multimodal fusion + anchor-based contrastive training and Stage-2 retrain."""
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def train_or_eval_model(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    pbar = tqdm(dataloader)
    for batch_id, data in enumerate(pbar):
        with torch.cuda.amp.autocast():
            if data is None:
                continue
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), video_inputs_2.cuda(), batch_labels.cuda()
            
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            video_hidden, video_logits = video_s(video_inputs)
            video_hidden_2, video_logits_2 = video_s_masked(video_inputs_2)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            
            pred_video_hidden, pred_visual_logits = fusion_visual(text_hidden, video_hidden_2, video_hidden)
            pred_hidden, pred_logits = fusion(text_hidden, pred_video_hidden, audio_hidden)
            
            input_orig = pred_hidden
            input_aug = None
            label = batch_labels
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
            else:
                loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
  
            pred = torch.argmax(anchor_scores[mask], dim=-1)

            preds.append(pred)
            labels.append(label)
            losses.append(loss.item())

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
                if batch_id % args.accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                sentiment_representations.append(loss_output.sentiment_representations)
                sentiment_labels.append(loss_output.sentiment_labels)

    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    max_cosine = loss_output.max_cosine

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    
    f1_scores = []

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7
        # n = 15

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine


def _forward(model, loss_function, input_orig, input_aug, label, device):
    input_ids = input_orig.to(device)
    label = label.to(device)
    mask = torch.ones(len(input_orig)).to(device)
    mask = mask > 0.5
    if model.training:
        log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
        loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    else:
        with torch.no_grad():
            log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
            loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    loss = loss_output.ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * loss_output.cl_loss

    return loss, loss_output, log_prob, label[mask], mask, anchor_scores

def retrain(model, loss_function, dataloader, epoch, device, args,
            optimizer=None, lr_scheduler=None, train=False):
    losses, ce_losses, preds, labels = [], [], [], []

    if train:
        model.train()
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    pbar = tqdm(dataloader)
    for batch_id, batch in enumerate(pbar):
        data, label = batch
        data = data.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.set_grad_enabled(train):
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    logits = model(data)
                    loss = loss_function(logits, label)
            else:
                logits = model(data)
                loss = loss_function(logits, label)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Stage2][Epoch {epoch+1}][Batch {batch_id}] invalid loss detected")
            print("data nan:", torch.isnan(data).any().item(), "inf:", torch.isinf(data).any().item())
            print("logits nan:", torch.isnan(logits).any().item(), "inf:", torch.isinf(logits).any().item())
            print("label min/max:", label.min().item(), label.max().item())
            raise ValueError("NaN/Inf loss in stage2 retrain")

        losses.append(loss.item())
        ce_losses.append(loss.item())

        pred = torch.argmax(logits, dim=-1)
        preds.append(pred.detach())
        labels.append(label.detach())

        if train:
            loss = loss / args.accumulation_step
            loss.backward()

            if (batch_id + 1) % args.accumulation_step == 0 or (batch_id + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    if len(preds) == 0:
        return float('nan'), float('nan'), [], [], float('nan'), []

    new_preds = []
    new_labels = []
    for i, label_batch in enumerate(labels):
        for j, l in enumerate(label_batch):
            if l != -1:
                new_labels.append(l.cpu().item())
                new_preds.append(preds[i][j].cpu().item())

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    f1_scores = []
    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    n = 6 if args.dataset_name in ['IEMOCAP'] else 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                pred_label.append(1 if new_preds[i] == class_id else 0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                true_label.append(0)
        if len(true_label) == 0 or len(pred_label) == 0:
            f1 = 0.0
        else:
            f1 = round(f1_score(true_label, pred_label, zero_division=0) * 100, 2)
        f1_scores.append(f1)

    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores