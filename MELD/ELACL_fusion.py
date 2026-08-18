"""Stage 1+2: ELACL anchor-contrastive training on top of HMF fusion for MELD."""
import os
import json
import numpy as np
import argparse
import random
from dataclasses import dataclass
import copy
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader, sampler, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *
from ELACL_loss import loss_function
from ELACL_trainer import train_or_eval_model, retrain
from tqdm import tqdm
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=8, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--dataset_name', default='MELD', type= str, help='dataset name, IEMOCAP or MELD or EmoryNLP')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0) 
    parser.add_argument("--temp", type=float, default=0.5) 

    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')
    parser.add_argument("--use_nearest_neighbour", type=bool, default=True)
    parser.add_argument("--anchor_path", type=str, default = './MELD/emo_anchors/roberta-large')
    parser.add_argument("--disable_emo_anchor", action='store_true')
    parser.add_argument("--disable_two_stage_training", type=bool, default=True)
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)
    parser.add_argument("--save_stage_two_cache", type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')
    parser.add_argument('--stage2_batch_size', type=int, default=32)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--teacher_ckpt', default='./MELD/save_model/teacher.bin', type=str)
    parser.add_argument('--audio_ckpt', default='./MELD/save_model/student_audio/total_student.bin', type=str)
    parser.add_argument('--visual_ckpt', default='./MELD/save_model/student_video2/total_student.bin', type=str)
    parser.add_argument('--visual_masked_ckpt', default='./MELD/save_model/student_video_masked2/total_student.bin', type=str)
    parser.add_argument('--fusion_visual_ckpt', default='./MELD/save_model/total_fusion_visual2.bin', type=str)
    parser.add_argument('--fusion_ckpt', default='./MELD/save_model/total_fusion2.bin', type=str)
    parser.add_argument('--save_dir', default='./MELD/save_model', type=str,
                        help='directory for Stage 1/Stage 2 checkpoints and selection metadata')
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

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    params = list(map(id, model.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        weight_decay = 0.01
        if id(param) in params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            args.ptmlr / 4 if id(param) in params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/MELD.Raw/'

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'
    
    train_pickle_path = data_path + 'train_faces_data.pkl'
    dev_pickle_path = data_path + 'dev_faces_data.pkl'
    test_pickle_path = data_path + 'test_faces_data.pkl'

    train_dataset = meld_dataset(preprocessing_pickle(train_path, train_pickle_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs_pickle)

    dev_dataset = meld_dataset(preprocessing_pickle(dev_path, dev_pickle_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs_pickle)

    test_dataset = meld_dataset(preprocessing_pickle(test_path, test_pickle_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs_pickle)

    save_path = args.save_dir

    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(args.teacher_ckpt))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load(args.audio_ckpt))
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load(args.visual_ckpt))
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    video_s_masked = Student_Video(video_model, clsNum)
    video_s_masked.load_state_dict(torch.load(args.visual_masked_ckpt))
    for para in video_s_masked.parameters():
        para.requires_grad = False
    video_s_masked = video_s_masked.cuda()
    video_s_masked.eval()


    '''fusion'''
    # hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.1, 3
    fusion_visual = ASF_visual(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion_visual.load_state_dict(torch.load(args.fusion_visual_ckpt))
    for para in fusion_visual.parameters():
        para.requires_grad = False
    fusion_visual = fusion_visual.cuda()
    fusion_visual.eval()

    # hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion.load_state_dict(torch.load(args.fusion_ckpt))
    for para in fusion.parameters():
        para.requires_grad = False
    fusion = fusion.cuda()
    fusion.eval()

    n_classes = 7

    stage1_model = CLModel(args, n_classes, fusion)
    stage1_model = stage1_model.cuda()
    stage1_model.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    max_grad_norm = args.max_grad_norm

    lr = args.lr
    num_training_steps = 1
    num_warmup_steps = 0
    optimizer = AdamW(get_paramsgroup(stage1_model.module if hasattr(stage1_model, 'module') else stage1_model))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
    best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    device = "cuda"
    best_acc = 0.
    best_valid_fscore = 0.
    best_stage1_epoch = None

    # stage 1
    for e in tqdm(range(training_epochs)):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine  = \
            train_or_eval_model(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, stage1_model, loss_function, train_loader, e, device, args, optimizer, lr_scheduler, True)
        lr_scheduler.step()
        valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _ = \
            train_or_eval_model(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, stage1_model, loss_function, dev_loader, e, device, args)
        all_fscore.append([valid_fscore])

        print( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,
            round(time.time() - start_time, 2)))

        if valid_fscore > best_valid_fscore:
            best_model = copy.deepcopy(stage1_model)
            best_valid_fscore = valid_fscore
            torch.save(stage1_model.state_dict(), save_path  + '/stage1_model_' + args.anchor_path.split('/')[-1] + '.pkl')
            best_stage1_epoch = e + 1
    
    print('finish stage 1 training !')

    stage1_ckpt_path = save_path + '/stage1_model_' + args.anchor_path.split('/')[-1] + '.pkl'
    stage1_model.load_state_dict(torch.load(stage1_ckpt_path))
    stage1_model.eval()
    _, _, _, _, best_stage1_test_fscore, _, _ = train_or_eval_model(
        model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, stage1_model,
        loss_function, test_loader, best_stage1_epoch or 0, device, args)
    print('Stage 1 validation-best test F-Score: ', best_stage1_test_fscore)
    with open(save_path + '/stage1_selection.json', 'x') as handle:
        json.dump({'checkpoint': stage1_ckpt_path, 'selection_split': 'validation',
                   'selection_metric': 'weighted_f1', 'best_epoch': best_stage1_epoch,
                   'best_validation_f1': best_valid_fscore,
                   'test_f1': best_stage1_test_fscore, 'seed': args.seed,
                   'test_evaluated_once_after_selection': True}, handle, indent=2)
    
    # stage 2
    torch.cuda.empty_cache()
    with torch.no_grad():
        stage1_model.load_state_dict(torch.load(stage1_ckpt_path))
        stage1_model.eval()
        anchors = stage1_model.map_function_anchor(stage1_model.emo_anchor)
        emb_train, emb_val, emb_test = [] ,[] ,[]
        label_train, label_val, label_test = [], [], []
        for batch_id, batch in enumerate(train_loader):
            if batch is None:
                continue
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch
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

            input_ids = input_orig.cuda()
            label = label.cuda()
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = stage1_model(input_ids, return_mask_output=True) 
            emb_train.append(masked_mapped_output.detach().cpu())
            label_train.append(label.cpu())

        emb_train = torch.cat(emb_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        for batch_id, batch in enumerate(dev_loader):
            if batch is None:
                continue
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch
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

            input_ids = input_orig.cuda()
            label = label.cuda()

            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = stage1_model(input_ids, return_mask_output=True) 
            emb_val.append(masked_mapped_output.detach().cpu())
            label_val.append(label.cpu())

        emb_val = torch.cat(emb_val, dim=0)
        label_val = torch.cat(label_val, dim=0)
        for batch_id, batch in enumerate(test_loader):
            if batch is None:
                continue
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, video_inputs_2, batch_labels = batch
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

            input_ids = input_orig.cuda()
            label = label.cuda()

            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = stage1_model(input_ids, return_mask_output=True) 
            emb_test.append(masked_mapped_output.detach().cpu())
            label_test.append(label.cpu())

        emb_test = torch.cat(emb_test, dim=0)
        label_test = torch.cat(label_test, dim=0)

    print("Embedding dataset built")

    all_fscore = []
    trainset = TensorDataset(emb_train, label_train)
    validset = TensorDataset(emb_val, label_val)
    testset = TensorDataset(emb_test, label_test)
    sampler = torch.utils.data.RandomSampler(
        trainset
    )

    stage2_batch_size = args.stage2_batch_size
    train_loader = DataLoader(trainset, batch_size=stage2_batch_size, shuffle=True, pin_memory=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=stage2_batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(testset, batch_size=stage2_batch_size, shuffle=False, num_workers=8)

    clf = Classifier(args, anchors).cuda()
    device = "cuda"
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
    best_valid_score = 0
    best_stage2_epoch = None
    for e in range(args.stage2_epochs):
        train_loss, train_ce_loss, train_acc, _, _, train_fscore, train_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).cuda(), train_loader, e, device, args, optimizer, train=True)
        
        valid_loss, valid_ce_loss,  valid_acc, _, _, valid_fscore, valid_detail_f1  = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).cuda(), valid_loader, e, device, args, optimizer, train=False)
        print( 'Epoch: {}, train_loss: {}, train_ce_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}'. \
                format(e + 1, train_loss, train_ce_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore))
        all_fscore.append([valid_fscore])
        
        if valid_fscore > best_valid_score:
            best_valid_score = valid_fscore
            # torch.save(clf.state_dict(), args.anchor_path.split('/')[-1] + '2.pkl')
            torch.save(clf.state_dict(),  save_path  + '/stage2_model_' + args.anchor_path.split('/')[-1] + '.pkl')
            best_stage2_epoch = e + 1
    stage2_ckpt_path = save_path  + '/stage2_model_' + args.anchor_path.split('/')[-1] + '.pkl'
    clf.load_state_dict(torch.load(stage2_ckpt_path))
    _, _, _, test_label, test_pred, best_stage2_test_fscore, test_detail_f1 = retrain(
        clf, nn.CrossEntropyLoss(ignore_index=-1).cuda(), test_loader,
        best_stage2_epoch or 0, device, args, optimizer, train=False)
    print('Stage 2 validation-best test F-Score: ', best_stage2_test_fscore)
    print(test_detail_f1)
    with open(save_path + '/stage2_selection.json', 'x') as handle:
        json.dump({'checkpoint': stage2_ckpt_path, 'selection_split': 'validation',
                   'selection_metric': 'weighted_f1', 'best_epoch': best_stage2_epoch,
                   'best_validation_f1': best_valid_score,
                   'test_f1': best_stage2_test_fscore, 'seed': args.seed,
                   'test_evaluated_once_after_selection': True}, handle, indent=2)

    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
