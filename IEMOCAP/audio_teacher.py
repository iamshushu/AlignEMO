"""Audio teacher training for the teacher-modality ablation (CMC revision, M2).

Trains the audio encoder (Student_Audio) as a *teacher* with plain cross-entropy
(no distillation), so it can act as the CMKD teacher in the swapped configuration
(audio -> text/visual) and be compared against the default text teacher.
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
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
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

def model_train(training_epochs, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore = float('-inf')
    best_epoch = 0
    ce_loss = nn.CrossEntropyLoss()
    for epoch in tqdm(range(training_epochs)):
        model_s.train()
        for i_batch, data in enumerate(train_dataloader):
            if data is None:
                continue
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
                batch_audio, batch_labels = batch_audio.cuda(), batch_labels.cuda()
                hidden_s, logit_s = model_s(batch_audio)
                loss_val = ce_loss(logit_s, batch_labels)

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

    model_s.load_state_dict(torch.load(os.path.join(save_path, 'total_teacher.bin')))
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
            batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
            batch_audio, batch_labels = batch_audio.cuda(), batch_labels.cuda()
            hidden_s, logit_s = model_s(batch_audio)
            pred_label = logit_s.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_teacher.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3

    """Dataset Loading"""

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

    save_audio = os.path.join(args.save_dir, "audio_teacher")

    print("###Save Path### ", save_audio)
    if not os.path.exists(save_audio):
        os.makedirs(save_audio)

    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''audio teacher model (trained standalone, no distillation)'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s = audio_s.cuda()

    """Training Setting"""
    training_epochs = args.epochs
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer_audio = torch.optim.AdamW(audio_s.parameters(), lr=lr)
    scheduler_audio = get_linear_schedule_with_warmup(optimizer_audio, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, audio_s, trainloader, devloader, testloader, optimizer_audio, scheduler_audio, max_grad_norm, scaler, save_audio)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
