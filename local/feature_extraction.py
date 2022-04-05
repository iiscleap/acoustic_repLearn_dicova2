#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021

@author: Team DiCOVA, IISC, Bangalore
"""
import argparse, configparser
import pickle
import librosa, os, random
import numpy as np
from utils_funcs import *

def compute_SAD(sig,fs,threshold=0.0001,sad_start_end_sil_length=100, sad_margin_length=50):

    sad_start_end_sil_length = int(sad_start_end_sil_length*1e-3*fs)
    sad_margin_length = int(sad_margin_length*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>threshold] = 1
    sad = np.zeros(sig.shape)
    for i in range(sample_activity.shape[1]):
        if sample_activity[0,i] == 1:
            sad[0,i-sad_margin_length:i+sad_margin_length] = 1
    sad[0,0:sad_start_end_sil_length] = 0
    sad[0,-sad_start_end_sil_length:] = 0
    return sad


def read_audio(file_path, sampling_rate):
    """
    try:
        fs=librosa.get_samplerate(file_path)
        s,_ = librosa.load(file_path,sr=sampling_rate)
        if np.mean(s)==0 or len(s)<1024:
            raise ValueError()
        # waveform level amplitude normalization
        s = s/np.max(np.abs(s))
        
        
    except:
        s = None
        print("Read audio failed for "+file_path)	
    """
    s,fs = torchaudio.load(file_path)
        
    return s, fs

def compute_mfcc(s,config):
    # Compute MFCC using librosa toolkit. 
    F = librosa.feature.mfcc(s,sr=int(config['default']['sampling_rate']),
                                n_mfcc=int(config['mfcc']['n_mfcc']),
                                n_fft = int(config['default']['window_size']),
                                hop_length = int(config['default']['window_shift']),
                                n_mels = int(config['mfcc']['n_mels']),
                                fmax = int(config['mfcc']['fmax']))

    features = np.array(F)
    if config['mfcc']['add_deltas'] in ['True','true','TRUE','1']:
        deltas = librosa.feature.delta(F)
        features = np.concatenate((features,deltas),axis=0)

    if config['mfcc']['add_delta_deltas'] in ['True','true','TRUE','1']:
        ddeltas = librosa.feature.delta(F,order=2)
        features = np.concatenate((features,ddeltas),axis=0)

    return features

def compute_logmelspec(s,config):
    # Compute MFCC using librosa toolkit. 
    F = librosa.feature.melspectrogram(s+1e-16,sr=int(config['default']['resampling_rate']),
                                n_fft = int(int(config['default']['window_size'])*44.1),
                                hop_length = int(int(config['default']['hop_length'])*44.1),
                                n_mels = int(config['logMelSpec']['n_mels']),
                                fmax = int(config['logMelSpec']['f_max']))
    F = np.log10(F+1e-16)
    features = np.array(F)
    if config['logMelSpec']['add_deltas'] in ['True','true','TRUE','1']:
        deltas = librosa.feature.delta(F)
        features = np.concatenate((features,deltas),axis=0)

    if config['logMelSpec']['add_delta_deltas'] in ['True','true','TRUE','1']:
        ddeltas = librosa.feature.delta(F,order=2)
        features = np.concatenate((features,ddeltas),axis=0)

    return features

def main(config, in_wav_list, out_folder):

    in_wav_list = open(in_wav_list).readlines()
    in_wav_list = [line.strip().split(" ") for line in in_wav_list]
    random.shuffle(in_wav_list)
    feats_list=[]
    for file_id, file_path in in_wav_list:
        out_file_name = out_folder+"/"+file_id + '.pkl' #+'_'+config['default']['feature_type']+'.pkl'
        if os.path.exists(out_file_name):  
            feats_list.append((file_id,out_file_name))
        #else:
            continue
        #file_path = file_path.split('..')[-1] 

        s,fs = read_audio(file_path,int(config['default']['resampling_rate']))
        if convertType(config['default']['pre_emphasis']):
            s = librosa.effects.preemphasis(s.numpy().reshape(-1))
            s = torch.from_numpy(s).reshape(1,-1)
        s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(config['default']['resampling_rate'])]])           
        s = s/torch.max(torch.abs(s))

        if s is None:
            continue
          
        sad = compute_SAD(s.numpy(), int(config['default']['resampling_rate']),threshold=1e-4)
        #sad = sad.numpy()
        ind=np.where(sad==1)
        s = s[ind]	# Equivalent to stripping the silence portions of the waveform
        if len(s)<1024:
            continue

        if config['default']['feature_type'] == 'mfcc':
            f = compute_mfcc(s,config)
        elif config['default']['feature_type'] == 'logMelSpec':
            f = compute_logmelspec(s.numpy(),config)
        else:
            raise ValueError("Need to implement the feature: "+config['default']['feature_type'])

        
        with open(out_file_name,'wb') as fp:
            pickle.dump(f,fp)
        feats_list.append((file_id,out_file_name))
    with open(out_folder+"/feats.scp","w") as f:
        for file_id,out_file_name in feats_list:
            f.write(file_id+" "+out_file_name+"\n")	

    print("Feature extraction completed for "+str(len(in_wav_list))+" files")
    print("Feature matrices saved to "+out_folder)
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c',required=True)
    parser.add_argument('--in_wav_list','-i',required=True)
    parser.add_argument('--out_folder','-o',required=False)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    main(config, args.in_wav_list, args.out_folder)
