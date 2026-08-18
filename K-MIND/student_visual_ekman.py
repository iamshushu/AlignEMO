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
from utils_ekman import *
from dataset_ekman import *
from model import *
from kmind_kd import *

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

def CE_Loss(pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)

    loss_val = ori_loss + logit_loss + feature_loss 
    return loss_val

def model_train(student_type, training_epochs, model_t, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore = float('-inf')
    best_epoch = 0
    model_t.eval()
    for epoch in tqdm(range(training_epochs)):
        model_s.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels = data
                if student_type == "visual":
                    batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_video.cuda(), batch_video_2.cuda(), batch_labels.cuda()
                    hidden_s, logit_s = model_s(batch_video)
                    hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)
                else:
                    batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_video.cuda(), batch_video_2.cuda(), batch_labels.cuda()
                    hidden_s, logit_s = model_s(batch_video_2)
                    hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)

                loss_val = CE_Loss(logit_s, logit_t, hidden_s, hidden_t, batch_labels)
                  
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
            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels = data
            
            if student_type == "visual":
                batch_video, batch_labels = batch_video.cuda(), batch_labels.cuda()
                hidden_s, logit_s = model_s(batch_video)
            else:
                batch_video_2, batch_labels = batch_video_2.cuda(), batch_labels.cuda()
                hidden_s, logit_s = model_s(batch_video_2)

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

    # text_model = "roberta-large"
    text_model = "klue/roberta-base"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/KMIND/'

    train_path = data_path + 'splits/train_ekman_0210.csv'
    dev_path = data_path + 'splits/dev_ekman_0210.csv'
    test_path = data_path + 'splits/test_ekman_0210.csv'

    train_pickle_path = data_path + 'faces_train_2_0210.pkl'
    dev_pickle_path = data_path + 'faces_dev_2_0210.pkl'
    test_pickle_path = data_path + 'faces_test_2_0210.pkl'

    train_dataset = kimind_dataset(preprocessing_pickle(train_path, train_pickle_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=video_batchs_pickle)

    dev_dataset = kimind_dataset(preprocessing_pickle(dev_path, dev_pickle_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=video_batchs_pickle)

    test_dataset = kimind_dataset(preprocessing_pickle(test_path, test_pickle_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=video_batchs_pickle)


    save_video = os.path.join('./K-MIND/save_model', "student_video_2")
    save_video_masked = os.path.join('./K-MIND/save_model', "student_video_masked_2")
    
    print("###Save Path### ", save_video)
    if not os.path.exists(save_video):
        os.makedirs(save_video)
  
    print("###Save Path### ", save_video_masked)
    if not os.path.exists(save_video_masked):
        os.makedirs(save_video_masked)

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
    video_s = Student_Video(video_model, clsNum)
    video_s = video_s.cuda()
    video_s.eval()

    video_s_masked = Student_Video(video_model, clsNum)
    video_s_masked = video_s_masked.cuda()
    video_s_masked.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer_video = torch.optim.AdamW(video_s.parameters(), lr=lr)
    scheduler_video = get_linear_schedule_with_warmup(optimizer_video, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    optimizer_video_masked = torch.optim.AdamW(video_s_masked.parameters(), lr=lr)
    scheduler_video_masked = get_linear_schedule_with_warmup(optimizer_video_masked, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()
    print('visual')
    model_train("visual", training_epochs, model_t, video_s, train_loader, dev_loader, test_loader, optimizer_video, scheduler_video, max_grad_norm, scaler, save_video)
    # print('visual_masked')
    # model_train("visual_masked", training_epochs, model_t, video_s_masked, train_loader, dev_loader, test_loader, optimizer_video_masked, scheduler_video_masked, max_grad_norm, scaler, save_video_masked)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
