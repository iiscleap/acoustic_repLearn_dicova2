import argparse, configparser
import csv
#from dicova_FFNet.models import MLP2L
#from dicova_FFNet.dataset_dicova import CoswaraDatasetFrameLevel
#from utils_dataset import custom_collate
import os
import sys
import io
import time
import numpy as np
import random
import torch
import pickle 
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from utils_funcs import *
from dataset_dicova import CoswaraDatasetFrameLevel
from utils_dataset import RawCoswaraDataset, custom_collate
from Net_learn_means_AcFB import AcFB, AcFB_FreeMeanVar, cnn4, Cnn14
from models import MLP2L
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io as sio


sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True) 

experiment = 'exp_LM_baselineFull_joint'
parser = argparse.ArgumentParser(description=experiment)

# ================== Input files and dirs ====================
parser.add_argument('--audio_category', default='speech', choices={'speech','cough','breathing'},
                    type=str, help='One of audiocategories from {speech,cough,breathing}')
parser.add_argument('--exp_path',default='./',
                    type=str, help='path to create results and checkpoint dirs')
parser.add_argument('--config','-c', required=True,
                        help='config file for train and val dataset')
parser.add_argument('--feat_config','-f', required=True,
                        help='config file for feature generation')
parser.add_argument('--model_config','-m',required=True)
parser.add_argument('--featsfil', required=True,
                         help='file containing all feature paths')
parser.add_argument('--trainfil', required=True, 
                     help='path to train csv file')
parser.add_argument('--valfil', required=True, 
                     help='path to val csv file')
parser.add_argument('--result_folder', required=True, 
                     help='Output folder to save results/checkpoint')

# ================   *********** =========================
parser.add_argument('--num_workers', default=4) 

parser.add_argument('--num_audio_channels',default=1,type=int,
                    help='#audio channels')
parser.add_argument('--num_freq_bin',default=64,type=int,
                    help='nuber of frequency bins')

parser.add_argument('--mean_var_norm', default=True)

parser.add_argument('--num_classes',default=2,type=int,
                    help='number of classes')
parser.add_argument('--batch_size',default=64,type=int,
                    help='batch size')

parser.add_argument('--num_epochs',default=30,type=int,
                    help='number of epochs')
parser.add_argument('--seed',default=1,type=int,
                    help='seed for torch.manual_seed()')
parser.add_argument('--num_files_load',default=100,type=int,
                    help='#files to load per split of dataset')

parser.add_argument('--net_activation',default='relu', choices=['tanh','relu'],
                    type=str, help='type activation for classifier')

#============  Optimizer and Scheduler ==============
parser.add_argument('--loss_type',default='BCE', choices=['CE','BCE'],
                    type=str, help='type of loss fn')
parser.add_argument('--optim_method',default='adam', choices=['sgd','adam'],
                    type=str, help='name of optimizer')
parser.add_argument('--max_lr',default=0.1,type=float,
                    help='initial LR for baseline architecture')
parser.add_argument('--acfb_lr',default=0.01,type=float,
                    help='initial LR for AcFB net')
parser.add_argument('--use_scheduler', default=True, action='store_true',
                    help='#files to load per split of dataset')
parser.add_argument('--scheduler', default='LRreduceOnPlateau',
                      help='#files to load per split of dataset')
parser.add_argument('--l2_regularization', action='store_true',
                    help='#If true, puts l2-regularization')
parser.add_argument('--l2_lambda', default=1e-4,
                    help='#If true, puts l2-regularization')



#============= Resume and AcFB train Modes ======================
parser.add_argument('--resume', dest='resume', default=False, action='store_true',
                    help='#files to load per split of dataset')

parser.add_argument('--freeze_acfb', action='store_true',
                    help='#If true, fixes the acfb at loading/initalization')
parser.add_argument('--mel_classifier', action='store_true',
                    help='classifier with mel features')
parser.add_argument('--learn_vars', action='store_true',
                    help='If true learn variances in the AcFB')
parser.add_argument('--last_model_path',default=None, type=str,
                    help='path to the last saved model')
parser.add_argument('--start_epoch',default=0, dest='start_epoch', type=int,
                    help='starting epoch number')
parser.add_argument('--best_loss',default=1e8,type=float,
                    help='best loss till current epoch')

args = parser.parse_args()
#print(args)
for arg in vars(args):
    print(arg,'=', getattr(args, arg))
cfg = open(args.config).readlines()
print(cfg)

# ============  dataset configs =================
config = configparser.ConfigParser()
config.read(args.config)

trds_args={}
for key in list(config['training_dataset'].keys()):
    val = config['training_dataset'][key]
    trds_args[key] = convertType(val)
print(trds_args)
vlds_args={}
for key in list(config['validation_dataset'].keys()):
    val = config['validation_dataset'][key]
    vlds_args[key] = convertType(val)
print(vlds_args)


feat_config = configparser.ConfigParser()
feat_config.read(args.feat_config)
feat_args={}
for key in list(feat_config['logMelSpec'].keys()):
    val = feat_config['logMelSpec'][key]
    feat_args[key] = convertType(val)
print(feat_args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = args.best_loss
start_epoch = args.start_epoch
current_lr = args.max_lr
args.use_cuda = torch.cuda.is_available()

if args.seed != 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)



""" Data """
print('===> Preparing data...')

file_list = open(args.featsfil).readlines()
file_list = [line.strip().split() for line in file_list]

file_paths = {}
for line in file_list:
    file_paths[line[0]]=line[1]

train_label_info = open(args.trainfil).readlines()
train_label_info = [line.strip().split() for line in train_label_info]

labels = {}
categories = ['n','p']
for fil,label in train_label_info:
    labels[fil]=categories.index(label)

val_label_info = open(args.valfil).readlines()
val_label_info = [line.strip().split() for line in val_label_info]

val_labels = {}
categories = ['n','p']
for fil,label in val_label_info:
    val_labels[fil]=categories.index(label)

#==================== Load Networks =========================

#net = Cnn14(classes_num=2)
net = MLP2L(input_dim = 3*feat_args['n_mels'], loss_type=args.loss_type,
            activation_type=args.net_activation)


if args.freeze_acfb:   # Only supports mel initialization for now
    acfb = AcFB(ngf=args.num_freq_bin, init='mel',
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate']))
        )
elif args.learn_vars: # Learn both mean and vars
    acfb = AcFB_FreeMeanVar(ngf=args.num_freq_bin, init='mel',
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate']))
        )

else:    # Learn only means
    acfb = AcFB(ngf=args.num_freq_bin, init='mel',
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate']))
        )




print(acfb)
print(net)

criterion = nn.CrossEntropyLoss()

if args.use_cuda:
    net.cuda()
    if acfb is not None:
        acfb.cuda()
    criterion.cuda()
    print('Using', torch.cuda.device_count(), 'GPUs.')
    #cudnn.benchmark = True  ## Enabling makes it faster training if input is of fixed size
    print('Using CUDA...')


if args.freeze_acfb:     # DON'T UPDATE AcFB
    if args.optim_method == 'sgd':
        optimizer = torch.optim.SGD([ 
                                {'params': net.parameters(),'lr': args.max_lr},
                                ], momentum=0.9, weight_decay=1e-6)

    elif args.optim_method == 'adam':
        optimizer = torch.optim.Adam([
                                {'params': net.parameters(),'lr': args.max_lr},
                                ],  
                                weight_decay=float(config['training']['weight_decay']))

else:

    if args.optim_method == 'sgd':
        optimizer = torch.optim.SGD([ {'params': acfb.parameters(), 'lr':args.acfb_lr},
                                {'params': net.parameters(),'lr': args.max_lr},
                                ], momentum=0.9, weight_decay=1e-6)
    
    elif args.optim_method == 'adam':
        optimizer = torch.optim.Adam([{'params': acfb.parameters(), 'lr':args.acfb_lr},
                                {'params': net.parameters(),'lr': args.max_lr},
                                ],  
                                weight_decay=float(config['training']['weight_decay']))


if args.use_scheduler: 
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer,1,
                                            T_mult=2,
                                            eta_min=args.max_lr*1e-4
                                            #last_epoch = -1
                                            #verbose=True
                                            )
    else:    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,
                                                        patience=5)

if args.last_model_path is not None:
    print(f'====> Loading model {args.last_model_path} ...')
    checkpoint = torch.load(args.last_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    acfb.load_state_dict(checkpoint['acfb_state_dict'])
    loss = checkpoint['loss'],
    #best_loss = checkpoint['best_loss']
    auc = checkpoint['auc'],
    epoch = checkpoint['epoch'],
    start_epoch = checkpoint['epoch'] + 1
    
    if args.use_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)



train_losses = []
val_losses = []
train_batchloss_list = []
val_batchloss_list = []
val_AUCs = []
train_AUCs = []
best_frameauc = 0
best_fileauc = 0
best_loss = 1e8

if args.freeze_acfb:
        
    val_loss, val_batch_losses, vlout_labels, vlref_labels = test(epoch, val_label_info, val_labels, net, acfb, args, 
            optimizer, scheduler, criterion, vlds_args)
elif args.mel_classifier:
        
    val_loss, val_batch_losses,vlout_labels, vlref_labels  = test_MelClassifier(epoch, val_label_info, val_labels, net, args, 
            optimizer, scheduler, criterion, vlds_args)
else:
        
    val_loss, val_batch_losses, vlout_labels, vlref_labels = testTorchDelta(epoch, val_label_info, val_labels, net, acfb, args, 
            optimizer, scheduler, criterion, vlds_args)

if not args.mel_classifier:

    vl_scores = scoring(vlref_labels, vlout_labels)
    file_outs, file_refs = test_fileLevel(epoch, val_label_info, val_labels, net, args, 
            vlds_args, acfb=acfb) #-----------
    
    file_scores = scoring(file_refs, file_outs)



print('frame level:\n:')
print(vl_scores)

print('file level:\n:')
print(file_scores)

print('scoring done!')



