import os
import sys
import time
import math
import torch
import numpy as np
import pandas as pd 
import torchaudio
import pickle
import soundfile as sf
import torch.nn.functional as F
import librosa
import random
from pdb import set_trace as bp
import tqdm
from utils_funcs import *




def test_fileLevel(epoch, test_label_info, test_labels, net, args, 
             vlds_args, acfb=None):
    net.eval()
    if acfb is not None:
        acfb.eval()

    test_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    batches_seen = 0
    batch_losses = []
    out_labels = []
    ref_labels = []


    file_list = open(args.featsfil).readlines()
    file_list = [line.strip().split() for line in file_list]
    file_paths = {}
    for line in file_list:
        file_paths[line[0]]=line[1]

    temp = test_label_info
    labels={}
    categories = ['n','p']

    for fil, label in temp:
        labels[fil]=categories.index(label)
    del temp

    for fil in list(labels.keys()):
        path = file_paths[fil]
        F = pickle.load(open(path,'rb')) 
        F = torch.from_numpy(F)
        label = labels.get(fil,None)
        #print(F.shape)

        F, label = F.float().cuda(), torch.tensor([label]).float().cuda()

        inputs = F.unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # 1,1,f,t
        with torch.no_grad():
            inputs = acfb(inputs)
            inputs_deltas = torchaudio.functional.compute_deltas(inputs, win_length=9) # mode='reflect', 9 make it same as librosa
            inputs_deltas_deltas = torchaudio.functional.compute_deltas(inputs_deltas, win_length=9)

            inputs =  torch.cat((inputs,inputs_deltas,inputs_deltas_deltas),2) # B,3ngf,T-8,1

            # MVN OVER THE CONCATED FEATURES
            if vlds_args['apply_mean_norm']:
                inputs = inputs - torch.mean(inputs, dim=-1,keepdim=True)
            if vlds_args['apply_var_norm']:
                inputs = inputs / (torch.std(inputs, dim=-1,keepdim=True) + 1e-8)

            inputs = inputs.squeeze(1).permute(0,2,1) # B,T,3ngf
            inputs = inputs.view(-1, inputs.shape[2]).unsqueeze(1)  # .,1,3ngf
            output = net(inputs)
            output = torch.sigmoid(output)
            output = output.reshape(-1).mean() # scalar

            out_labels.append(output.detach().cpu().item())
            ref_labels.append(label.cpu().item())

    return out_labels, ref_labels

def split_train(model,train_lines, train_labels, optimizer, epoch, args, trds_args, feat_config, scheduler, augment=False, debug=False):
    
    total_trLoss = 0
    total_batches_seen = 0


    num_split = int(len(train_lines)/args.num_files_load)
    if num_split==0: num_split=1; print(f'num of split can not be zero, continuing with num_split=1') 

    print(f'Train dataset splitted into {num_split} parts','.'*10,'\n')
    
    idxs = np.arange(len(train_lines))
    np.random.shuffle(idxs)
    #train_lines = map(np.array, train_lines)
    train_lines = np.array(train_lines)
    train_lines = train_lines[idxs]
    #train_labels = train_labels[idxs]

    iters = len(train_lines) // args.batch_size + 1
    print('Total batches in the dataset:', iters*num_split)

    for i in range(num_split):
        if (i+1)*args.num_files_load <= len(train_lines):
            inlines = train_lines[i*args.num_files_load:(i+1)*args.num_files_load]
        else: 
            inlines = train_lines[i*args.num_files_load:len(train_lines)]
            
        # DEFINE DATALOADERS
        split_trDataset = RawCoswaraDataset_V2({'feat_config': feat_config,
                        'file_list': args.featsfil,  
                        'inlines': inlines,
                        'augment': augment,
                        'augdir': args.augdir,
                        'shuffle': True,
                        'dataset_args':trds_args,
                        'augtypes':['noise','time','pitch','shift']
                            })

        split_trData_loader = torch.utils.data.DataLoader(split_trDataset, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=custom_collate)
        if i == 0: print("No of batches per train split: ",len(split_trData_loader))
        avg_trEpochLoss = train(model, split_trData_loader, optimizer, epoch, args, scheduler, 
                        total_batches_seen, iters, splitId=i, debug=False)
        total_trLoss += avg_trEpochLoss * len(split_trData_loader)
        total_batches_seen += len(split_trData_loader)
        
    return total_trLoss/ total_batches_seen 



def train(model, data_loader, optimizer, epoch, args, scheduler, batches_seen, iters, splitId=1,  debug=False):
    ''' Function to train for an epoch '''
    if args.acfb_preTrained:
        model.acfb.eval()
    else:
        model.acfb.train()
    model.classifier.train()
    if args.use_relWt:
        model.relWt1.train()

    
    total_train_loss = 0
    for batch_id, (feats,labels) in enumerate(data_loader):
        if len(feats) == 1:
            continue
        if args.use_cuda:
            feats  = [x.float().cuda() for x in feats] #, labels.cuda()
            labels = [x.cuda() for x in labels]
        optimizer.zero_grad()
        # Zero the gradients before each batch
        if args.use_mixup:
            outputs, targets_a, targets_b, lam = model(feats, labels, mode='train')
            loss_a = lam * model.classifier.criterion_mixup(torch.div(outputs, args.temp),targets_a).reshape(-1,)
            loss_b = (1-lam) * model.classifier.criterion_mixup(torch.div(outputs, args.temp),targets_b).reshape(-1,)
            #loss = torch.mean(lam * model.classifier.criterion_mixup(torch.div(outputs, args.temp),targets_a).reshape(-1,) 
            #    +(1-lam) * model.classifier.criterion_mixup(torch.div(torch.stack(outputs), args.temp),targets_b).reshape(-1,))
            loss = torch.mean(loss_a + loss_b)

        else:
            outputs, targets = model(feats, labels)
            loss = model.classifier.criterion(outputs, targets)

        if args.l2_regularization:
            params1 = torch.cat([x.view(-1) for x in model.acfb.parameters()])
            params2 = torch.cat([x.view(-1) for x in model.classifier.parameters()]) 
            l2_loss1 = float(args.l2_lambda) * torch.norm(params1, 2)
            l2_loss2 = float(args.l2_lambda) * torch.norm(params2, 2) 
            loss = loss + l2_loss1 + l2_loss2

        # Forward pass through the net and compute loss
        loss.backward()
        # Backward pass and get gradients
        optimizer.step()
        if args.scheduler == 'cosine':
            scheduler.step(epoch + (batches_seen+batch_id) / iters)
        # Update the weights
        if debug and np.isnan(loss.detach().item()):
            print('found a nan at {}'.format(batch_id))
        total_train_loss += loss.detach().item()
        progress_bar(batch_id + 1, len(data_loader),
                        'Loss: %.3f | split_id: %.1f | Acc: %.3f%% (%d/%d)'
                        %(total_train_loss/(batch_id +1),splitId,
                        0.,0,1))
        
    return total_train_loss/(batch_id+1)

def validate(model, data_loader, args):
    ''' Do validation and compute loss 
        The scores and labels are also stacked to facilitate AUC computation
    '''
    model.eval()
    y_scores, y_val = [], []
    total_val_scores = 0
    num_egs = 0
    total_val_loss = 0
    with torch.no_grad():
        for batch_id, (feats,labels) in enumerate(data_loader):
            if len(feats) == 1:
                continue
            if args.use_cuda:
                feats = [x.float().cuda() for x in feats]
                labels = [x.cuda() for x in labels]

            output, _ = model(feats, labels,mode='val')
            # Predict the score
            val_loss = model.classifier.criterion(output, torch.stack(labels))	
            # Compute loss
            y_val.extend([item.cpu().numpy() for item in labels])
            y_scores.extend([torch.sigmoid(item).cpu().numpy() for item in output]) 
            #Get proba
            total_val_scores +=  val_loss.detach().item()*len(labels)
            num_egs+=len(labels)
            total_val_loss += val_loss

            progress_bar(batch_id + 1, len(data_loader),
                        'Loss: %.3f | split_id: %.1f | Acc: %.3f%% (%d/%d)'
                        %(total_val_loss/(batch_id +1),1,
                        0.,0,1))

    return total_val_scores/num_egs, y_val, y_scores #AUC, TPR, TNR

