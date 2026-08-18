"""Text student knowledge-distillation for the teacher-modality ablation (M2).

Swapped configuration: the *audio* encoder is the frozen teacher and the *text*
encoder is the student distilled from it (mirror of student_audio.py, which
distills audio from a text teacher). Same CMKD loss (CE + logit + feature).
"""
import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *
from iemocap_kd import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--teacher_ckpt', default='./IEMOCAP/save_model/audio_teacher/total_teacher.bin', type=str, help='path to (audio) teacher checkpoint')
    parser.add_argument('--save_dir', default='./IEMOCAP/save_model', type=str, help='directory to save model checkpoints')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CE_Loss(pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)

    loss_val = ori_loss + 0.1*logit_loss + feature_loss
    return loss_val

def model_train(training_epochs, model_t, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore = float('-inf')
    best_epoch = 0
    model_t.eval()
    for epoch in tqdm(range(training_epochs)):
        model_s.train()
        for i_batch, data in enumerate(train_dataloader):
            if data is None:
                continue
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
                batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_audio.cuda(), batch_labels.cuda()
                # student = text, teacher = audio (swapped vs student_audio.py)
                hidden_s, logit_s = model_s(batch_input_tokens, batch_attention_masks)
                hidden_t, logit_t = model_t(batch_audio)
                loss_val = CE_Loss(logit_s, logit_t, hidden_s, hidden_t, batch_labels)

            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        model_s.eval()
        dev_pred_list, dev_label_list = evaluation(model_s, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(model_s, save_path)

    model_s.load_state_dict(torch.load(os.path.join(save_path, 'total_student.bin')))
    test_pred_list, test_label_list = evaluation(model_s, test_dataloader)
    _, _, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
    print(f"validation-best test_score : {test_fbeta}")

def evaluation(model_s, dataloader):
    model_s.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            if data is None:
                continue
            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
            batch_input_tokens, batch_attention_masks, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_labels.cuda()
            hidden_s, logit_s = model_s(batch_input_tokens, batch_attention_masks)
            """Calculation"""
            pred_label = logit_s.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_student.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3

    """Dataset Loading"""

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"

    data_path = './dataset/IEMOCAP_full_release/'

    train_path = data_path + 'IEMOCAP_train.csv'
    dev_path = data_path + 'IEMOCAP_dev.csv'
    test_path = data_path + 'IEMOCAP_test.csv'

    train_dataset = iemocap_dataset(preprocessing(train_path))
    trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=audio_batchs)

    dev_dataset = iemocap_dataset(preprocessing(dev_path))
    devloader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=audio_batchs)

    test_dataset = iemocap_dataset(preprocessing(test_path))
    testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=audio_batchs)

    save_text = os.path.join(args.save_dir, "student_text")

    print("###Save Path### ", save_text)
    if not os.path.exists(save_text):
        os.makedirs(save_text)

    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load (audio teacher, frozen)'''
    model_t = Student_Audio(audio_model, clsNum, init_config)
    model_t.load_state_dict(torch.load(args.teacher_ckpt))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model (text)'''
    text_s = Teacher_model(text_model, clsNum)
    text_s = text_s.cuda()

    """Training Setting"""
    training_epochs = args.epochs
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer_text = torch.optim.AdamW(text_s.parameters(), lr=lr)
    scheduler_text = get_linear_schedule_with_warmup(optimizer_text, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, model_t, text_s, trainloader, devloader, testloader, optimizer_text, scheduler_text, max_grad_norm, scaler, save_text)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
