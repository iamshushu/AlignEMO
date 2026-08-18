"""Train/eval loops for Stage 1 (HMF) and Stage 2 (ELACL classifier) on IEMOCAP."""
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
            
            pred_video_hidden, pred_visual_logits = fusion_visual(text_embedding=text_hidden, visual=video_hidden_2, visual_2=video_hidden)

            pred_hidden, pred_logits = fusion(text_embedding=text_hidden, visual=pred_video_hidden, acoustic=audio_hidden)
            
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

def retrain(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, ce_losses, preds, labels = [], [], [], []
    pbar = tqdm(dataloader)
    for batch in pbar:
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                log_prob = model(data) 
        
        loss = loss_function(log_prob, label)
        losses.append(loss.item())
        ce_losses.append(loss.item())
        pred = torch.argmax(log_prob, dim = -1)
        preds.append(pred)
        labels.append(label)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
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
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    f1_scores = []

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

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
 
    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores
