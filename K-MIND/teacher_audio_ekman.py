"""Stage 1A: Whisper audio teacher training for K-MIND (Ekman 7-way)."""
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoProcessor, Data2VecAudioModel
from transformers import AutoImageProcessor, TimesformerModel
import gc

from preprocessing import *
from utils_ekman_whisper import *
from dataset_ekman import *
from model import *
from kmind_kd import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--save_dir', default='./K-MIND/save_model/student_audio_2_whisper',
                        type=str, help='directory for the validation-best checkpoint')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--train_csv', default='./dataset/KMIND/splits/train_ekman_0210.csv')
    parser.add_argument('--dev_csv', default='./dataset/KMIND/splits/dev_ekman_0210.csv')
    parser.add_argument('--test_csv', default='./dataset/KMIND/splits/test_ekman_0210.csv')
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

def CELoss(pred_outs, labels):
    # loss = nn.CrossEntropyLoss()
    loss = nn.CrossEntropyLoss(ignore_index=-1) 
    loss_val = loss(pred_outs, labels)
    return loss_val


def model_train(student_type, training_epochs, model_t, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore = float('-inf')
    best_epoch = 0
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
                hidden_s, logit_s = model_s(batch_audio)
                loss_val = CELoss(logit_s, batch_labels)
                  
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)  
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        model_s.eval()   
        dev_pred_list, dev_label_list = evaluation(student_type, model_s, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(model_s, save_path)

    model_s.load_state_dict(torch.load(os.path.join(save_path, 'total_student_ekman.bin')))
    test_pred_list, test_label_list = evaluation(student_type, model_s, test_dataloader)
    _, _, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
    print(f"validation-best test_score : {test_fbeta}")

def evaluation(student_type, model_s, dataloader):
    model_s.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):      
            if data is None:
                continue      
            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
            batch_audio, batch_labels = batch_audio.cuda(), batch_labels.cuda()
            hidden_s, logit_s = model_s(batch_audio)


            """Calculation"""    
            
            pred_label = logit_s.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_student_ekman.bin')) 

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3

    """Dataset Loading"""

    text_model = "klue/roberta-base"
    audio_model = "openai/whisper-large-v3"
    # audio_model = "facebook/data2vec-audio-base-960h"
   
    data_path = './dataset/KMIND/'

    train_path = args.train_csv
    dev_path = args.dev_csv
    test_path = args.test_csv
    
    # num_workers=16
    train_dataset = kimind_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=audio_batchs_whisper)

    dev_dataset = kimind_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=audio_batchs_whisper)

    test_dataset = kimind_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=audio_batchs_whisper)

    save_audio = args.save_dir
   
    print("###Save Path### ", save_audio)
    if not os.path.exists(save_audio):
        os.makedirs(save_audio)
   
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = None

    '''student model'''
    # audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s = Student_Audio_Whisper(audio_model, clsNum, init_config)
    audio_s = audio_s.cuda()
    audio_s.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_loader)*training_epochs
    num_warmup_steps = len(train_loader)
    optimizer_audio = torch.optim.AdamW(audio_s.parameters(), lr=lr) 
    scheduler_audio = get_linear_schedule_with_warmup(optimizer_audio, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train("audio", training_epochs, model_t, audio_s, train_loader, dev_loader, test_loader, optimizer_audio, scheduler_audio, max_grad_norm, scaler, save_audio)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
