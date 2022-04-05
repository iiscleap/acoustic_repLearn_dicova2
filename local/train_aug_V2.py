
import argparse, configparser
import csv
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
from pdb import set_trace as bp
from scoring import *
from dataset_dicova import CoswaraDatasetFrameLevel
from utils_trains import * 
from Net_learn_means_AcFB import AcFB, AcFB_FreeMeanVar, cnn4, Cnn14
from models import MLP2L, getJointNet, LSTMClassifier
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io as sio
from utils_funcs import *


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
parser.add_argument('--augdir', default=None, type=str, 
                     help='dir for aug files')

# ================   *********** =========================
parser.add_argument('--num_workers', default=4) 

parser.add_argument('--num_audio_channels',default=1,type=int,
                    help='#audio channels')
parser.add_argument('--num_freq_bin',default=64,type=int,
                    help='nuber of frequency bins')

parser.add_argument('--mean_var_norm', action='store_true')

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
parser.add_argument('--wdecay',default=1e-6,type=float,
                    help='weight decay for optimizer')
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
parser.add_argument('--acfb_preTrained', action='store_true',
                    help='If true learn variances in the AcFB')
parser.add_argument('--random_net', action='store_true',
                     help="If true initialize classifier with random weights")
parser.add_argument('--LBaseline', action='store_true',
                    help='If true then train for dicova2 baseline')
                    
# ========================== RELEVANCE ==================================

parser.add_argument('--use_relWt', action='store_true',
                    help='If true train with relevance weighting')
parser.add_argument('--relevance_type', choices=['bandWt', 'adaptiveWt'], default='adaptiveWt',
                    help='If true train with types of relevance weighting relevance weighting')
parser.add_argument('--relContext', type=int, default=51,
                     help='context_length for relevance net')
parser.add_argument('--use_skipConn', action='store_true',
                    help='If true, adds skip-connection in the relevance pipeline')

parser.add_argument('--deltas', action='store_true')
parser.add_argument('--use_mixup', action='store_true')
parser.add_argument('--temp', type=float, default=0.01)


parser.add_argument('--acfb_init', default='mel', choices=['mel', 'invMel', 'linear','rand1','rand2'],
                      type=str)
parser.add_argument('--last_model_path',default=None, type=str,
                    help='path to the last saved model')
parser.add_argument('--start_epoch',default=0, dest='start_epoch', type=int,
                    help='starting epoch number')
parser.add_argument('--best_loss',default=1e8,type=float,
                    help='best loss till current epoch')

parser.add_argument('--summary_file',default='score_summary.txt',type=str,
                    help='Summary file for scores')

args = parser.parse_args()
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

# MODEL CONFIG
model_args = {'input_dimension':3*args.num_freq_bin}
if args.model_config != 'None':
    temp = configparser.ConfigParser()
    temp.read(args.model_config)
    for key in temp['default'].keys():
        model_args[key]=convertType(temp['default'][key])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = args.best_loss
start_epoch = args.start_epoch
current_lr = args.max_lr
args.use_cuda = torch.cuda.is_available()

if args.seed != 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)



""" Data Lists """
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
net = LSTMClassifier(model_args)

if args.freeze_acfb:   # Only supports mel initialization for now
    acfb = AcFB(ngf=args.num_freq_bin, init='mel',
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        seed = args.seed)

elif args.learn_vars: # Learn both mean and vars
    acfb = AcFB_FreeMeanVar(ngf=args.num_freq_bin, init=args.acfb_init,
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        seed = args.seed)
 

else:    # Learn only means
    acfb = AcFB(ngf=args.num_freq_bin, init=args.acfb_init,
        win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
        seed = args.seed)



model = getJointNet(acfb, net, args)
print(model)

criterion = nn.BCEWithLogitsLoss()

if args.use_cuda:
    net.cuda()
    if acfb is not None:
        acfb.cuda()
    criterion.cuda()
    if model is not None:
        model.cuda()
    print('Using', torch.cuda.device_count(), 'GPUs.')
    print('Using CUDA...')

if args.freeze_acfb or args.acfb_preTrained:     # DON'T UPDATE AcFB
    model.acfb.means.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),args.max_lr,
                                 weight_decay=args.wdecay)
                                

else:
    
    optimizer = torch.optim.Adam([{'params': model.acfb.parameters(), 'lr':args.acfb_lr,'weight_decay':0},
                            {'params': model.classifier.parameters(),'lr': args.max_lr,'weight_decay':args.wdecay},
                            ],  
                            )

    if args.use_relWt:
        optimizer = torch.optim.Adam([{'params': model.acfb.parameters(), 'lr':args.acfb_lr,'weight_decay':0},
                            {'params': model.classifier.parameters(),'lr': args.max_lr,'weight_decay':args.wdecay},
                            {'params': model.relWt1.parameters(),'lr': args.max_lr,'weight_decay':args.wdecay},
                            ]
                            )

print(optimizer)

if args.use_scheduler: 
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer,1,
                                            T_mult=2,
                                            eta_min=args.max_lr*1e-4
                                            )
    else:    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,
                                                        patience=5)

if args.last_model_path is not None and args.last_model_path not in [ "None", "", " "]:
    print('====> Resuming from checkpoint...')
    print(f'======> Loading model {args.last_model_path}')
    checkpoint = torch.load(args.last_model_path)
    acfb.load_state_dict(checkpoint['acfb_state_dict'])
    if not args.random_net:
        net.load_state_dict(checkpoint['model_state_dict'])

    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

    
# Result dir
if not os.path.isdir(args.exp_path + '/results'):
    os.mkdir(args.exp_path + 'results')

logname=(args.exp_path + 'results/log_'+ experiment +'_' + net.__class__.__name__+'_'+str(args.seed) + '.csv')

validation_dataset = RawCoswaraDataset_V2({'feat_config': feat_config,
                    'file_list': args.featsfil,  
                    'inlines': val_label_info,
                    'augment': False,
                    'augdir': args.augdir,
                    'shuffle': False,  
                    'dataset_args':vlds_args,
                    'augtypes': None
                        })

                    
train_losses = []
val_losses = []
train_batchloss_list = []
val_batchloss_list = []
val_AUCs = []
train_AUCs = []
train_file_AUCs = []
val_file_AUCs = []
best_frameauc = 0
best_fileauc = 0
best_loss = 1e8

meansdir = args.result_folder + 'figs/meanplots/'
if not os.path.exists(meansdir):
    os.makedirs(meansdir)

augment = True if args.augdir not in [None,"","none","None"] else False

if config['training']['lr_scheme']=='ExponentialLR':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=float(config['training']['learning_rate_decay']))
elif config['training']['lr_scheme']=='ReduceLROnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=float(config['training']['learning_rate_decay']))
elif config['training']['lr_scheme']=='custom':
    min_val_loss = 1e8
    gt_min_val_cnt = 0
else:
    raise ValueError('Unknown LR scheme')

val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
val_loss, y_val, y_scores  = validate(model,val_data_loader, args)
print("Initial val loss: {}".format(val_loss))



#==================================  Epoch Trainng ===========================================
#=============================================================================================

t0 = time.time()
for epoch in range(args.num_epochs):
    
    print("\n","*"*50, f" Epoch {epoch} starting ","*"*50,"\n")
    
    if args.acfb_preTrained:
        # PLOT INITIAL MEANS
        if epoch == 0:
            save_means_plot(acfb.means.detach().cpu().numpy(), -0.01, 'inf', 
                    figsdir=meansdir,fs=44100, ngf=args.num_freq_bin)  

        val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
        
        train_loss = split_train(model, train_label_info, labels, optimizer, epoch, args, trds_args, feat_config, scheduler, debug=False)
        val_loss, y_val, y_scores  = validate(model,val_data_loader, args)



    elif args.LBaseline or args.learn_vars:

        if epoch == 0:
            if args.learn_vars:
                save_means_plot(acfb.means.detach().cpu().numpy(), -1, 'inf', 
                    figsdir=meansdir,stds=acfb.stds.detach().cpu().numpy(),fs=44100, ngf=args.num_freq_bin)
            else:
                save_means_plot(acfb.means.detach().cpu().numpy(), -1, 'inf', 
                    figsdir=meansdir,fs=44100, ngf=args.num_freq_bin)  

        val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
        
        train_loss = split_train(model, train_label_info, labels, optimizer, epoch, args, trds_args, feat_config, scheduler, augment=augment, debug=False)
        val_loss, y_val, y_scores  = validate(model,val_data_loader, args)

        if not os.path.exists(meansdir):
            os.makedirs(meansdir)
        meanfile = meansdir + f'means-epoch-{epoch}.mat'
        sio.savemat(meanfile, mdict={'data':acfb.means.cpu().detach().numpy()})
        if args.learn_vars:
            varfile = meansdir + f'vars-epoch-{epoch}.mat'
            sio.savemat(varfile, mdict={'data':acfb.stds.cpu().detach().numpy()})


    else:
        # PLOT INITIAL MEANS
        if epoch == 0:
            save_means_plot(acfb.means.detach().cpu().numpy(), -1, 'inf', 
                    figsdir=meansdir,fs=44100, ngf=args.num_freq_bin)  

        val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
        
        train_loss = split_train(model, train_label_info, labels, optimizer, epoch, args, trds_args, feat_config, scheduler, debug=False)
        val_loss, y_val, y_scores  = validate(model,val_label_info, args)

        meansdir = args.result_folder + 'figs/meanplots/'
        if not os.path.exists(meansdir):
            os.makedirs(meansdir)
        meanfile = meansdir + f'means-epoch-{epoch}.mat'
        sio.savemat(meanfile, mdict={'data':acfb.means.cpu().detach().numpy()})

    if config['training']['lr_scheme'] == 'custom' and args.scheduler != 'cosine':
        if val_loss <=min_val_loss:
            min_val_loss=val_loss
            gt_min_val_cnt = 0
        else:
            gt_min_val_cnt+=1
            if gt_min_val_cnt>2:
                min_val_loss=val_loss
                for gid,g in enumerate(optimizer.param_groups):
                    g['lr']*=float(config['training']['learning_rate_decay'])
                new_lr = g['lr']
                print('='*20+ f" LR updated to {new_lr} "+'='*20)
                if g['lr']<1e-8: break
    elif args.scheduler == 'cosine':
         print("-"*20 +" LR updated to: {} ".format(scheduler.get_last_lr())+"-"*20) 
    else:
        raise ValueError(f"{args.scheduler} scheduler not implemented!")

    val_losses.append(val_loss)
    train_losses.append(train_loss)
    scores = score(y_val, y_scores)
    val_auc = scores['AUC']

    val_AUCs.append(val_auc)


    # =========== PLOT MEAN LOSSES ======================
    lossdir = args.result_folder + 'figs/lossplots/'
    if args.acfb_preTrained:
        lossdir = args.result_folder + 'figs/ClsLossplots/'
        
    plot_fig(train_losses, val_losses,
            dir=lossdir, title='loss',
            leg1='train',leg2='val',
            save=True, filename='loss.png')


    #============= PLOT AUCs =============================
    plot_fig(val_AUCs, dir=lossdir,
            title='frame val AUC',leg1='auc',
            save=True, filename='auc.png')

    save_checkpoint(val_loss, val_auc, epoch, args, net,
                            acfb, model, optimizer, scheduler)


    # ============ PLOT MEANS ========================
    if not args.mel_classifier and not args.acfb_preTrained:
        if args.learn_vars:
            save_means_plot(acfb.means.detach().cpu().numpy(), epoch, val_loss, 
                    figsdir=meansdir,stds=acfb.stds.detach().cpu().numpy(),fs=44100, ngf=args.num_freq_bin)

        else:
            save_means_plot(acfb.means.detach().cpu().numpy(), epoch, val_loss, 
                    figsdir=meansdir,fs=44100, ngf=args.num_freq_bin)
        plt.close()   
    

    if val_loss < best_loss:

        best_loss = val_loss
        best_loss_epoch = epoch
        save_checkpoint(val_loss, val_auc, epoch, args, net, 
                    acfb, model, optimizer, scheduler, name='best_loss') 

        
    if val_auc > best_frameauc:
        best_frameauc = val_auc
        best_auc_epoch = epoch
        save_checkpoint(val_loss, val_auc, epoch, args, net, 
                    acfb, model, optimizer, scheduler, name='best_frame_auc')



save_checkpoint(val_loss, val_auc, epoch, args, net, 
                acfb, model, optimizer, scheduler, name='final')



 



