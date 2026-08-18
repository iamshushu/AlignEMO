"""K-MIND full tri-modal fusion training (text + visual-fusion + audio) for Ekman 7-way ERC."""
import glob
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

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from preprocessing import *
from utils_ekman import *
from dataset_ekman import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
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
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        for i_batch, data in enumerate(train_dataloader):
            if data is None:
                continue
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), video_inputs_2.cuda(), batch_labels.cuda()
                
                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)
                video_hidden_2, video_logits_2 = video_s_masked(video_inputs_2)

                pred_hidden, pred_logits = fusion_visual(text_hidden, video_hidden_2, video_hidden)

                pred_hidden_all, pred_logits_all = fusion(text_hidden, pred_hidden, audio_hidden)

                loss_val = CELoss(pred_logits_all, batch_labels)
            
            
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(fusion, save_path)

    fusion.load_state_dict(torch.load(os.path.join(save_path, 'total_fusion_ekman_grid.bin')))
    test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, test_dataloader)
    _, _, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
    print(f"validation-best test_score : {test_fbeta}")

def evaluation(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, dataloader):
    fusion.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            if data is None:
                continue
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), video_inputs_2.cuda(), batch_labels.cuda()
                                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            video_hidden, video_logits = video_s(video_inputs)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden_2, video_logits_2 = video_s_masked(video_inputs_2)
                
            pred_hidden, pred_logits = fusion_visual(text_hidden, video_hidden_2, video_hidden)
            pred_hidden_all, pred_logits_all = fusion(text_hidden, pred_hidden, audio_hidden)

            
            """Calculation"""    

            pred_label = pred_logits_all.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_fusion_ekman_grid.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
    text_model = "klue/roberta-base"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/KMIND/'

    train_path = data_path + 'splits/train_ekman_0210.csv'
    dev_path = data_path + 'splits/dev_ekman_0210.csv'
    test_path = data_path + 'splits/test_ekman_0210.csv'

    train_pickle_path = data_path + 'faces_train_2_0210.pkl'
    dev_pickle_path = data_path + 'faces_dev_2_0210.pkl'
    test_pickle_path = data_path + 'faces_test_2_0210.pkl'



    train_dataset = kimind_dataset(preprocessing_pickle(train_path, train_pickle_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs_pickle)

    dev_dataset = kimind_dataset(preprocessing_pickle(dev_path, dev_pickle_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs_pickle)

    test_dataset = kimind_dataset(preprocessing_pickle(test_path, test_pickle_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs_pickle)

    save_path = os.path.join('./K-MIND/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./K-MIND/save_model/teacher_ekman.bin'), strict=False)
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load('./K-MIND/save_model/student_audio_2/total_student_ekman.bin')) 
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load('./K-MIND/save_model/student_video_2/total_student_ekman.bin')) 
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    video_s_masked = Student_Video(video_model, clsNum)
    video_s_masked.load_state_dict(torch.load('./K-MIND/save_model/student_video_masked_2/total_student_ekman.bin')) 
    for para in video_s_masked.parameters():
        para.requires_grad = False
    video_s_masked = video_s_masked.cuda()
    video_s_masked.eval()

    '''fusion'''
    # hidden_size, beta_shift, dropout_prob, num_head = 768, 2e-1, 0.2, 4
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion_visual = ASF_visual(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion_visual.load_state_dict(torch.load('./K-MIND/save_model/total_fusion_visual_ekman_grid.bin')) 
    for para in fusion_visual.parameters():
        para.requires_grad = False
    fusion_visual = fusion_visual.cuda()
    fusion_visual.eval()

    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion = fusion.cuda()
    fusion.eval()


    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
