import argparse, configparser
import pickle
import logging
import torch
import torch.nn as nn
from scoring import *
import numpy as np
import matplotlib.pyplot as plt
from models import *
from utils_funcs import *
import torchaudio
from Net_learn_means_AcFB import AcFB, AcFB_FreeMeanVar, cnn4, Cnn14
from models import MLP2L, getJointNet, LSTMClassifier
import tqdm 

def main(modelfil, model_args, file_list,outfil,config, feat_config, feat_args, args):
    ''' Script to do inference using trained model
    config, feature_config: model coonfiguration and feature configuration files
    modelfil: trained model stored as a .mdl file
    file_list: list of files as in "<id> <file-path>" format
    outfil: output file, its content will be "<id> <probability-score>"
    '''

    net = LSTMClassifier(model_args)

    if args.learn_vars:
        acfb = AcFB_FreeMeanVar(ngf=args.num_freq_bins, init='mel',
            win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            seed = args.seed)
    else:
        acfb = AcFB(ngf=args.num_freq_bins, init='mel',
            win_len = int(float(feat_config['default']['window_size']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            hop_len = int(float(feat_config['default']['hop_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            filt_h = int(float(feat_args['filter_length']) * 1e-3 * float(feat_config['default']['resampling_rate'])),
            seed = args.seed)


    print(acfb)
    print(net)
    

    model = getJointNet(acfb, net, args)
    checkpoint = torch.load(modelfil, map_location='cuda')
    model.acfb.load_state_dict(checkpoint['acfb_state_dict'])
    model.classifier.load_state_dict(checkpoint['model_state_dict'])
    if args.use_relWt:
        model.relWt1.load_state_dict(checkpoint['relev_state_dict'])
    #loss = checkpoint['loss']
    #best_loss = checkpoint['best_loss']
    #start_epoch = checkpoint['epoch'] + 1

    # Load model, use CPU for inference
    #model = torch.load(modelfil,map_location='cpu')  ## load our model ------------
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    save_means_plot(model.acfb.means.detach().cpu().numpy(), -100000, 'inf', 
                    figsdir='./',fs=44100, ngf=args.num_freq_bins) 

    # Feature extractor
    #FE = feature_extractor(feature_config['default']) ## use frame feature extractor
    FE = FrameFeature_extractor(feat_config) ## Doesn't perform VAD

    # Loop over all files
    file_list = open(file_list).readlines()
    file_list = [line.strip().split() for line in file_list]

    feats_file = args.featsfil #'data/breathing_vad.scp'
    feats_lines=open(feats_file,'r').readlines()
    paths={}
    for line in feats_lines:
        key, path = line.strip().split(" ")
        paths[key]=path

    scores={}
    for fileId, _ in tqdm.tqdm(file_list):
        
        # Prepare features
        path = paths[fileId]
        F = FE.extract(path)	# F should be data frames: 
        F = torch.from_numpy(F)
        F = F.float()
        feat = F.to(device)

        # Input mode
        seg_mode = config['training_dataset'].get('mode','file')
        if seg_mode=='file':
            feat = [feat]
        elif seg_mode=='segment':
            segment_length = int(config['training_dataset'].get('segment_length',300))
            segment_hop = int(config['training_dataset'].get('segment_hop',10))
            feat = [feat[i:i+segment_length,:] for i in range(0,max(1,F.shape[0]-segment_length),segment_hop)]
        else:
            raise ValueError('Unknown eval model')
        with torch.no_grad():
            #output = model.predict_proba(feat) # output, labels = model(feat, labels)---------------------
            # check shapes here
            output, _ = model(feat, feat)
            output = [torch.sigmoid(x) for x in output]

        # Average the scores of all segments from the input file
        scores[fileId]= sum(output)[0].cpu().item()/len(output)

    # Write output scores
    with open(outfil,'w') as f:
        for item in scores:
            f.write(item+" "+str(scores[item])+"\n")

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfil','-m',required=True)
    parser.add_argument('--config','-c',required=True)
    parser.add_argument('--feat_config','-f',required=True)
    parser.add_argument('--file_list','-i',required=True)
    parser.add_argument('--featsfil',required=True)
    parser.add_argument('--outfil','-o',required=True)	
    parser.add_argument('--model_config',required=True)
    parser.add_argument('--acfb_preTrained', action='store_true')
    parser.add_argument('--learn_vars', action='store_true')
    parser.add_argument('--mean_var_norm', action='store_true')
    parser.add_argument('--deltas', action='store_true')
    parser.add_argument('--use_relWt', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--use_skipConn', action='store_true')
    parser.add_argument('--temp', type=int,default=0.01)
    parser.add_argument('--relContext', type=int,default=51)
    parser.add_argument('--relevance_type', choices=['bandWt', 'adaptiveWt'], default='adaptiveWt',
                    help='If true train with time types of relevance weighting')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_freq_bins',type=int, default=64)
    args = parser.parse_args()
    print(f"using {args.modelfil}")
    print(f"writng to {args.outfil}")

    config = configparser.ConfigParser()
    config.read(args.config)

    trds_args={}
    for key in list(config['training_dataset'].keys()):
        val = config['training_dataset'][key]
        trds_args[key] = convertType(val)
    #print(trds_args)
    vlds_args={}
    for key in list(config['validation_dataset'].keys()):
        val = config['validation_dataset'][key]
        vlds_args[key] = convertType(val)
    #print(vlds_args)


    feat_config = configparser.ConfigParser()
    feat_config.read(args.feat_config)
    feat_args={}
    for key in list(feat_config['logMelSpec'].keys()):
        val = feat_config['logMelSpec'][key]
        feat_args[key] = convertType(val)
    print(feat_args)

    # MODEL CONFIG
    model_args = {'input_dimension':3*args.num_freq_bins}
    if args.model_config != 'None':
        temp = configparser.ConfigParser()
        temp.read(args.model_config)
        for key in temp['default'].keys():
            model_args[key]=convertType(temp['default'][key])

    main(args.modelfil, model_args, args.file_list, args.outfil, config, feat_config, feat_args, args)
