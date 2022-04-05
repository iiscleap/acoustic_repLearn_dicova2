import subprocess
import numpy as np
import torch.nn as nn
import torch, torchaudio
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import time
import random
import sys, os
#from utils_dataset import RawCoswaraDataset,RawCoswaraDataset_V2, BaseCoswaraDataset, custom_collate
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.fft
import torchaudio.transforms
import librosa
import pickle 
import tqdm
from pdb import set_trace as bp
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    
            
def enframe(x, winlen, hoplen):
    """ receives an 1d array and divides it into frames"""

    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")

    x = np.pad(x, int(winlen//2), mode='reflect')

    n_frames = 1 + np.int(np.floor((len(x)-winlen)/float(hoplen)))
    #n_frames = 1 + np.int(np.floor(len(x)/float(hoplen)))
    
    xf = np.zeros((n_frames, winlen))
    for ii in range(n_frames):
        xf[ii] = x[ii * hoplen : ii * hoplen + winlen]

    #xf = xf.transpose() 
    return xf


class RawCoswaraDataset(torch.utils.data.Dataset):
    def __init__(self,args):
		
        self.file_list = args['file_list']
        #self.label_file = args['label_file']
        self.inlines = args['inlines']
        self.dataset_args = args['dataset_args']  # replace dataset args with args
        self.augmentaion_args = None #args.get('augmentation_args',None)
        self.shuffle = args.get('shuffle',False)
        #self.device = args['device']
        self.augtypes = args['augtypes']
        if self.augtypes is None:
            self.augtypes = ['noise','time'] #,'pitch','shift']

        self.mode = self.dataset_args['mode'] #----- add to args
        #self.mode = self.dataset_args['mode']
        if self.mode == 'segment':        
            self.segment_length = self.dataset_args['segment_length']
            self.segment_hop = self.dataset_args['segment_hop']
        self.oversampling = self.dataset_args.get('oversampling',False)
        if self.oversampling:
            self.oversampling_factor = self.dataset_args.get('oversampling_factor',-1)
        #self.apply_mean_norm = self.dataset_args.get('apply_mean_norm',True)
        #self.apply_var_norm = self.dataset_args.get('apply_var_norm',True)

        self.augment=args['augment']
        self.augdir = args['augdir']
        if self.augmentaion_args:
            if self.augmentaion_args['mode']=='masking':
                self.augment=True
                self.freq_mask=torchaudio.transforms.FrequencyMasking(self.augmentaion_args['freq_mask_param'])
                self.time_mask=torchaudio.transforms.TimeMasking(self.augmentaion_args['time_mask_param'])

        self.generate_examples()

    
    def do_oversample(self,F,label,egs):
        for i in range(self.oversampling_factor):	
            nF=int((0.8+0.2*np.random.rand())*F.shape[0])
            start=max(0,int(np.random.randint(0,F.shape[0]-nF,1)))
            #egs.append((F[start:start+nF,:].to(self.device),torch.FloatTensor([label]).to(self.device)))
            egs.append((F[start:start+nF,:], torch.FloatTensor([label])))
            #egs.append((F, torch.FloatTensor([label]))) 
        return egs




    def generate_examples(self):

		#%%
        file_list = open(self.file_list).readlines()
        file_list = [line.strip().split() for line in file_list]
        file_paths = {}
        for line in file_list:
            file_paths[line[0]]=line[1]
		#%%
        #temp = open(self.label_file).readlines()
        temp = self.inlines
        #temp = [line.strip().split() for line in temp]

        
        labels={}
        categories = ['n','p']

        for fil, label in temp:
            labels[fil]=categories.index(label)
        del temp

        if self.oversampling and self.oversampling_factor!=0:
            l = np.array(list(labels.values()))
            if np.sum(np.where(l==1)[0]) !=0:
                l = int(len(np.where(l==0)[0])/len(np.where(l==1)[0]))-1
                self.oversampling_factor=l
            

	
        egs = []
        tot_skipped = 0
        #bp()
        for fil in tqdm.tqdm(list(labels.keys())):
            path = file_paths[fil]
            
            F = pickle.load(open(path,'rb')) 
            F = torch.from_numpy(F)
            #if F.shape[1] != 192:
            #    F = F.permute(1,0)
            #if self.apply_mean_norm: F = F - torch.mean(F,dim=0)
            #if self.apply_var_norm: F = F / torch.std(F,dim=0)
                

            label = labels.get(fil,None)
            #egs.append( (F.to(self.device),torch.FloatTensor([label]).to(self.device)))
            egs.append( (F,torch.FloatTensor([label])))
            #print(fil)
            
            if label==1 and self.oversampling:
                
                egs = self.do_oversample(F,label, egs)
            
            if self.augment:
                
                for augtype in self.augtypes:
                    augpath = os.path.join(self.augdir, fil+'_'+augtype+'.pkl')
                    try:
                        F = pickle.load(open(augpath,'rb'))
                        F = torch.from_numpy(F)
                        egs.append((F,torch.FloatTensor([label])))
                    except:
                        #print(f'skipping {augpath}')
                        tot_skipped += 1
                        continue

                    if label==1 and self.oversampling:
                        egs = self.do_oversample(F,label, egs)          

        print(f'skipped {tot_skipped} augmented files...')   
        if self.mode=='file':
            egs=egs
        elif self.mode=='segment':
            fegs=[]
            for F,L in egs:
                start_pt=0;end_pt=min(F.shape[0],self.segment_length)
                while end_pt<=F.shape[0]:
                    fegs.append((F[start_pt:end_pt,:],L))
                    start_pt+=self.segment_hop;end_pt=start_pt+self.segment_length
            egs=fegs
        else:
            raise ValueError("Unknown mode")
        #print(len(egs))

        
        if self.augment:
            e1=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                eg = (F,l)
                e1.append(eg)
            e2=[]
            for eg in egs:
                F,l = eg
                F = self.time_mask(F)
                eg = (F,l)
                e2.append(eg)
            e3=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                F = self.time_mask(F)
                eg = (F,l)
                e3.append(eg)
            egs.extend(e1)
            egs.extend(e2)
            egs.extend(e3)
    
        if self.shuffle: random.shuffle(egs)
        self.egs=egs
        #print(len(egs))


    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        feat, label = self.egs[idx]

        return feat, label

class RawCoswaraDataset_V2(torch.utils.data.Dataset):
    def __init__(self,args):
		
        self.feat_config  = args['feat_config'] # after config.read
        self.file_list = args['file_list']
        #self.label_file = args['label_file']
        self.inlines = args['inlines']
        self.dataset_args = args['dataset_args']  # replace dataset args with args
        self.augmentaion_args = None #args.get('augmentation_args',None)
        self.shuffle = args.get('shuffle',False)
        #self.device = args['device']
        self.augtypes = args['augtypes']
        if self.augtypes is None:
            self.augtypes = ['noise','time'] #,'pitch','shift']

        self.mode = self.dataset_args['mode'] #----- add to args
        #self.mode = self.dataset_args['mode']
        if self.mode == 'segment':        
            self.segment_length = self.dataset_args['segment_length']
            self.segment_hop = self.dataset_args['segment_hop']
        self.oversampling = self.dataset_args.get('oversampling',False)
        if self.oversampling:
            self.oversampling_factor = self.dataset_args.get('oversampling_factor',-1)

        self.augment=args['augment']
        self.augdir = args['augdir']
        if self.augmentaion_args:
            if self.augmentaion_args['mode']=='masking':
                self.augment=True
                self.freq_mask=torchaudio.transforms.FrequencyMasking(self.augmentaion_args['freq_mask_param'])
                self.time_mask=torchaudio.transforms.TimeMasking(self.augmentaion_args['time_mask_param'])

        self.generate_examples()

    
    def do_oversample(self,F,label,egs):
        for i in range(self.oversampling_factor):	
            nF=int((0.8+0.2*np.random.rand())*F.shape[0])
            start=max(0,int(np.random.randint(0,F.shape[0]-nF,1)))
            #egs.append((F[start:start+nF,:].to(self.device),torch.FloatTensor([label]).to(self.device)))
            egs.append((F[start:start+nF,:], torch.FloatTensor([label])))
            #egs.append((F, torch.FloatTensor([label]))) 
        return egs




    def generate_examples(self):

		#%%
        file_list = open(self.file_list).readlines()
        file_list = [line.strip().split() for line in file_list]
        file_paths = {}
        for line in file_list:
            file_paths[line[0]]=line[1]
		#%%
        
        temp = self.inlines
        

        
        labels={}
        categories = ['n','p']

        for fil, label in temp:
            labels[fil]=categories.index(label)
        del temp

        if self.oversampling and self.oversampling_factor!=0:
            l = np.array(list(labels.values()))
            if np.sum(np.where(l==1)[0]) !=0:
                l = int(len(np.where(l==0)[0])/len(np.where(l==1)[0]))-1
                self.oversampling_factor=l
            

	
        egs = []
        tot_skipped = 0
        #bp()
        FE = FrameFeature_extractor(self.feat_config)
        for fil in tqdm.tqdm(list(labels.keys())):
            path = file_paths[fil]
           
            #bp() 
            #F = pickle.load(open(path,'rb')) 
            #F = torch.from_numpy(F)
            F = FE.extract(path)  # data_frames
            if F is None:
                continue
            F = torch.from_numpy(F)
                

            label = labels.get(fil,None)
            #egs.append( (F.to(self.device),torch.FloatTensor([label]).to(self.device)))
            egs.append( (F,torch.FloatTensor([label])))
            #print(fil)
            
            if label==1 and self.oversampling:
                
                egs = self.do_oversample(F,label, egs)
            
            if self.augment:
                
                for augtype in self.augtypes:
                    
                    augment = augment_with_audiomentations(augtype)
                    augsig  = augment.do_augment(path, fs=44100)
                    dummy_augpath=""
                    F = FE.extract(dummy_augpath, sig=augsig,fs=44100)
                    if F == None:
                        continue
                    F = torch.from_numpy(F)
                    egs.append((F,torch.FloatTensor([label])))

                    if label==1 and self.oversampling:
                        egs = self.do_oversample(F,label, egs)          

        #print(f'skipped {tot_skipped} augmented files...')   
        if self.mode=='file':
            egs=egs
        elif self.mode=='segment':
            fegs=[]
            for F,L in egs:
                start_pt=0;end_pt=min(F.shape[0],self.segment_length)
                while end_pt<=F.shape[0]:
                    fegs.append((F[start_pt:end_pt,:],L))
                    start_pt+=self.segment_hop;end_pt=start_pt+self.segment_length
            egs=fegs
        else:
            raise ValueError("Unknown mode")
        #print(len(egs))

       
        if self.shuffle: random.shuffle(egs)
        self.egs=egs
        #print(len(egs))


    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        feat, label = self.egs[idx]

        return feat, label


class BaseCoswaraDataset(torch.utils.data.Dataset):
    def __init__(self,args):
		
        self.file_list = args['file_list']
        #self.label_file = args['label_file']
        self.inlines = args['inlines']
        self.dataset_args = args['dataset_args']  # replace dataset args with args
        self.augmentaion_args = None #args.get('augmentation_args',None)
        self.shuffle = args.get('shuffle',False)
        self.subsample = self.dataset_args.get('subsampling_factor', 10)
        self.n_mels = 64
        self.sr = 44100
        self.deltas = False
        self.calc_mel = False
        #self.device = args['device']

        self.mode = self.dataset_args['mode'] #----- add to args
        #self.mode = self.dataset_args['mode']
        if self.mode == 'segment':        
            self.segment_length = self.dataset_args['segment_length']
            self.segment_hop = self.dataset_args['segment_hop']
        self.oversampling = self.dataset_args.get('oversampling',False)
        if self.oversampling:
            self.oversampling_factor = self.dataset_args.get('oversampling_factor',-1)
        self.apply_mean_norm = self.dataset_args.get('apply_mean_norm', True)
        self.apply_var_norm = self.dataset_args.get('apply_var_norm', True)

        self.augment=False
        if self.augmentaion_args:
            if self.augmentaion_args['mode']=='masking':
                self.augment=True
                self.freq_mask=torchaudio.transforms.FrequencyMasking(self.augmentaion_args['freq_mask_param'])
                self.time_mask=torchaudio.transforms.TimeMasking(self.augmentaion_args['time_mask_param'])

        self.generate_examples()



    def generate_examples(self):

		#%%
        file_list = open(self.file_list).readlines()
        file_list = [line.strip().split() for line in file_list]
        file_paths = {}
        for line in file_list:
            file_paths[line[0]]=line[1]
		#%%
        #temp = open(self.label_file).readlines()
        temp = self.inlines
        #temp = [line.strip().split() for line in temp]

        
        labels={}
        categories = ['n','p']

        for fil, label in temp:
            labels[fil]=categories.index(label)
        del temp

		#%%
        egs = []
        for fil in list(labels.keys()):
            path = file_paths[fil]
            F = pickle.load(open(path,'rb'))

            win_len = F.shape[1]

            if self.calc_mel:
             
                F = np.fft.fft(F, axis=1)
                F = F[:,:win_len//2+1].transpose()
                F = np.abs(F)**2
                F = librosa.feature.melspectrogram(S=F,sr=self.sr, n_fft=2048,
                        n_mels=self.n_mels,fmin=0,fmax=self.sr/2)
                F = librosa.amplitude_to_db(F, ref=1.0) / 2   # to maintain mul by 10
                #F = torch.from_numpy(F)
                F = F.transpose() #n_mels, t
                
            if F.shape[1] != 3 * self.n_mels:
                F = F.transpose()

            if self.apply_mean_norm: F = F - np.mean(F,axis=0)
            if self.apply_var_norm: F = F / (np.std(F,axis=0) + 1e-8)

            if self.deltas:
                F_delta = librosa.feature.delta(F, axis=0)
                F_delta_delta = librosa.feature.delta(F_delta, axis=0)
                F = np.concatenate((F,F_delta,F_delta_delta), axis=1)

            label = labels.get(fil,None)
            #egs.append( (F.to(self.device),torch.FloatTensor([label]).to(self.device)))
            label_all = np.vstack([label]*F.shape[0]).reshape(-1,1)
            egs.append( np.concatenate((F, label_all),axis=1))

        egs = np.vstack(egs)
        if self.shuffle: 
            random.shuffle(egs)
        egs = egs[::self.subsample]

        if self.oversampling and self.oversampling_factor!=0:
            ind = np.where(egs[:,-1]==1)[0]
            n_positives = len(ind)
            n_negatives = egs.shape[0] - n_positives
            positive_samples = egs[egs[:,-1]==1]
            if n_positives != 0:
                self.oversampling_factor = int(n_negatives/n_positives) - 1
                for i in range(self.oversampling_factor):
                    egs = np.concatenate((egs, positive_samples),axis=0)
            np.random.shuffle(egs)

        
            
        if self.mode=='file':
            egs=egs
        elif self.mode=='segment':
            fegs=[]
            for F,L in egs:
                start_pt=0;end_pt=min(F.shape[0],self.segment_length)
                while end_pt<=F.shape[0]:
                    fegs.append((F[start_pt:end_pt,:],L))
                    start_pt+=self.segment_hop;end_pt=start_pt+self.segment_length
            egs=fegs
        else:
            raise ValueError("Unknown mode")
        #print(len(egs))
        if self.augment:
            e1=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                eg = (F,l)
                e1.append(eg)
            e2=[]
            for eg in egs:
                F,l = eg
                F = self.time_mask(F)
                eg = (F,l)
                e2.append(eg)
            e3=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                F = self.time_mask(F)
                eg = (F,l)
                e3.append(eg)
            egs.extend(e1)
            egs.extend(e2)
            egs.extend(e3)
        
        self.egs=egs
        #print(len(egs))


    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        feat, label = self.egs[idx,:-1], self.egs[idx,-1]

        return feat, label


def custom_collate(batch):
    inputs = [t[0] for t in batch]
    targets = [t[1] for t in batch]
    return (inputs, targets)


def to_dict(filename):
	''' Convert a file with "key value" pair data to dictionary with type conversion'''
	data = open(filename).readlines()
	D = {}
	for line in data:
		key,val=line.strip().split()
		try:
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				pass
		D[key] = val
	return D
    


def compute_SAD(sig,fs,threshold=0.0001,sad_start_end_sil_length=20, sad_margin_length=50):

    sad_start_end_sil_length = int(sad_start_end_sil_length*1e-3*fs)
    sad_margin_length = int(sad_margin_length*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>threshold] = 1
    sad = np.zeros(sig.shape)

    N, idx  = sample_activity.shape[1], 0

    for i in range(sample_activity.shape[1]):
        if sample_activity[0,i] == 1:
            sad[0,i-sad_margin_length:i+sad_margin_length] = 1
    #bp()
    
    sad[0,0:sad_start_end_sil_length] = 0
    sad[0,-sad_start_end_sil_length:] = 0
    return sad

#%%
class feature_extractor():
    def __init__(self,args):

        self.args=args	
        self.vad_threshold = args.get('threshold',1e-4)
        self.resampling_rate = int(self.args['resampling_rate'])
        if self.args['feature_type'] == 'MFCC':
            self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate, 
                                                                n_mfcc=int(self.args['n_mfcc']), 
                                                                melkwargs={
                                                                    'n_fft': int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
                                                                    'n_mels': int(self.args['n_mels']), 
                                                                    'f_max': int(self.args['f_max']), 
                                                                    'hop_length': int(float(self.args['hop_length'])*1e-3*self.resampling_rate)})
        elif self.args['feature_type'] in ['MelSpec','logMelSpec'] :
            self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
                                                                    n_fft= int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
                                                                    n_mels= int(self.args['n_mels']), 
                                                                    f_max= int(self.args['f_max']), 
                                                                    hop_length= int(float(self.args['hop_length'])*1e-3*self.resampling_rate))
        else:
            raise ValueError('Feature type not implemented')

    def extract(self,filepath):

        s,fs = torchaudio.load(filepath)
        s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(self.resampling_rate)]])           
        s = s/torch.max(torch.abs(s))
        sad = compute_SAD(s.numpy(),self.resampling_rate,threshold=self.vad_threshold)
        s = s[np.where(sad==1)]     
        F = self.feature_transform(s)
        if self.args['feature_type'] == 'logMelSpec': 
            F = torchaudio.functional.amplitude_to_DB(F,multiplier=10,amin=1e-10,db_multiplier=0)
        return F.T

class FrameFeature_extractor():
    def __init__(self,args):

        self.args=args	
        self.vad_threshold = 1e-4 #args.get('threshold',1e-4)
        self.resampling_rate = int(self.args['default']['resampling_rate'])
        if self.args['default']['feature_type'] == 'MFCC':
            self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate, 
                                                                n_mfcc=int(self.args['n_mfcc']), 
                                                                melkwargs={
                                                                    'n_fft': int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
                                                                    'n_mels': int(self.args['n_mels']), 
                                                                    'f_max': int(self.args['f_max']), 
                                                                    'hop_length': int(float(self.args['hop_length'])*1e-3*self.resampling_rate)})
        elif self.args['default']['feature_type'] in ['MelSpec','logMelSpec'] :
            self.feat_args = self.args['logMelSpec']
            self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
                                                                    n_fft= int(float(self.args['default']['window_size'])*1e-3*self.resampling_rate), 
                                                                    n_mels= int(self.feat_args['n_mels']), 
                                                                    f_max= int(self.feat_args['f_max']), 
                                                                    hop_length= int(float(self.args['default']['hop_length'])*1e-3*self.resampling_rate))
        else:
            raise ValueError('Feature type not implemented')

    def extract(self,filepath,sig=None,fs=44100):

        if isinstance(sig, (np.ndarray, torch.Tensor)):
            s = sig
            if not isinstance(s, torch.Tensor):
                s = torch.from_numpy(s)
        else:
            s,fs = torchaudio.load(filepath)
	    
        if s.shape[0] > 1:  # Convert to mono
            s  = s[0,:].reshape(1,-1)
        if convertType(self.args['default']['pre_emphasis']):
            s = librosa.effects.preemphasis(s.numpy().reshape(-1))
            s = torch.from_numpy(s).reshape(1,-1)
        s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(self.resampling_rate)]])           
        s = s/torch.max(torch.abs(s))
        
        
        # CALCULATE VAD 
        sad = compute_SAD(s.numpy(),self.resampling_rate,threshold=self.vad_threshold)
        s = s[np.where(sad==1)]    
        if len(s) == 0: return None 
        n_frames = 1 + np.int(np.floor((len(s)-int(float(self.args['default']['window_size']))/int(float(self.args['default']['hop_length'])))))
            

        data_frames = enframe(s,int(float(self.args['default']['window_size'])*1e-3*self.resampling_rate),
                    int(float(self.args['default']['hop_length'])*1e-3*self.resampling_rate))

        return data_frames


def convertType(val):
    def subutil(val):
        try: 
            val = int(val)
        except:
            try:
                val = float(val)
            except:
                if val in ['True', 'TRUE', 'true']:
                    val = True
                elif val in ['False','FALSE','false']:
                    val = False
                elif val in ['None']:
                    val = None
                else:
                    val = val
        return val

    if ',' in val:
        val = val.split(',')
        val = [subutil(item) for item in val]
    else:
        val = subutil(val)
    return val

class augment_with_audiomentations():
    def __init__(self, augtype):

        self.augtype = augtype

        if augtype == 'noise':
            
            self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
        
            ])
        elif augtype == 'time':
            self.augment = Compose([    
                TimeStretch(min_rate=0.8, max_rate=1.25, p=1),      
            ])

        elif augtype == 'pitch':
            self.augment = Compose([
                PitchShift(min_semitones=-4, max_semitones=4, p=1),
            ])

        elif augtype == 'shift':
            self.augment = Compose([
                Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
            ])
        else:
            raise ValueError("Augmentation not implemented.")
    
    def do_augment(self, path, fs=44100, samples=None):
        if samples is not None:
            augsig = self.augment(samples=samples, sample_rate=fs)
        else:
            sig, fs = torchaudio.load(path)
            augsig = self.augment(samples=sig.numpy(), sample_rate=fs)
        return augsig

        
def get_freegpu():
    A,_= subprocess.Popen('nvidia-smi --format=csv,noheader --query-gpu=utilization.gpu,memory.used,index',shell=True,stdout=subprocess.PIPE).communicate()
    A = A.decode('utf-8').strip().split('\n')
    indices =  [int(item.split(",")[-1]) for item in A]
    order = np.argsort([int(item.split(' ')[0]) for item in A])
    return indices[order[0]]


def score(refs, sys_outs, sys_outs_oneHot=True):

    thresholds = np.arange(0,1,0.0001)   

    #%%
    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((1,len(thresholds)))
    FP = np.zeros((1,len(thresholds)))
    TN = np.zeros((1,len(thresholds)))
    FN = np.zeros((1,len(thresholds)))
    keyCnt=-1
    for sys_score,ref_label in zip(sys_outs, refs): # Repeat for each recording
        keyCnt+=1   
        """
        if sys_outs_oneHot:
            sys_labels = sys_score 
        """   
        sys_labels = (sys_score>=thresholds)*1	# System label for a range of thresholds as binary 0/1
        gt = ref_label
        
        ind = np.where(sys_labels == gt) # system label matches the ground truth
        if gt==1:	# ground-truth label=1: True positives 
            TP[0,ind]+=1
        else:		# ground-truth label=0: True negatives
            TN[0,ind]+=1
        ind = np.where(sys_labels != gt) # system label does not match the ground truth
        if gt==1:	# ground-truth label=1: False negatives
            FN[0,ind]+=1
        else:		# ground-truth label=0: False positives 
            FP[0,ind]+=1
            
    total_positives = sum(refs)	# Total number of positive samples
    total_negatives = len(refs)-total_positives # Total number of negative samples
    
    TP = np.sum(TP,axis=0)	# Sum across the recordings
    FP = np.sum(FP,axis=0)
    TN = np.sum(TN,axis=0)
    FN = np.sum(FN,axis=0)    
    
    TPR = TP/total_positives	# True positive rate: #true_positives/#total_positives
    FPR = FP/total_negatives	# False positive rate: #falsee_positives/#total_negatives
    FNR = FN/total_positives	# False negative rate: #false_negatives/#total_positives
    TNR = TN/total_negatives	# True negative rate: #true_negatives/#total_negatives

    Accuracy = (TP+TN)/(total_positives+total_negatives) # fraction of correct classifications
    Balanced_Accuracy = (TPR+TNR)/2	# fraction of correct classifications balanced for class distribution
    AUC = auc( FPR, TPR )    	# AUC 

    Precision = TP / (TP+FP+0.0001)	# Precision 
    Recall = TP/ (TP+FN+0.0001)		# Recall
    
    Fscore = 2*Precision*Recall/(Precision+Recall+0.0001)	# Fscore
    
    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except AUC) is a array for different threshold values

    scores={'TPR':TPR,
            'FPR':FPR,
            'TNR':TNR,
            'FNR':FNR,
            'Accuracy':Accuracy,
            'Balanced_Accuracy':Balanced_Accuracy,
            'AUC':AUC,
            'Precision':Precision,
            'Recall':Recall,
            'Fscore':Fscore,
            'thresholds':thresholds}
    return scores


# def scoring(refs,sys_outs,out_file=None,specificities_chosen=[0.5,0.95]):
#     """
#     inputs::
#     refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
#     sys_outs: a txt file with a list of scores (probability of being covid positive) for each wav-fileid in the format: <id> <score>
#     threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC
    
#     outputs::
        
#     """    

#     thresholds=np.arange(0,1,0.0001)

#     scores1 = score(refs, sys_outs) 
#     AUC = scores1['AUC']
#     TPR = scores1['TPR']
#     TNR = scores1['TNR']

#     specificities=[]
#     sensitivities=[]

#     decision_thresholds = []
#     for specificity_threshold in specificities_chosen:
#         ind = np.where(TNR>specificity_threshold)[0]
#         sensitivities.append( TPR[ind[0]])
#         specificities.append( TNR[ind[0]])
#         decision_thresholds.append( thresholds[ind[0]])


#     # pack the performance metrics in a dictionary to save & return
#     # Each performance metric (except AUC) is a array for different threshold values
#     # Specificity at 90% sensitivity
#     scores={'TPR':TPR,
#             'FPR':1-TNR,
#             'AUC':AUC,
#             'sensitivity':sensitivities,
#             'specificity':specificities,
#             'operatingPts':decision_thresholds,
#             'thresholds':thresholds}

#     if out_file != None:
#         with open(out_file,"wb") as f: pickle.dump(scores,f)
#         with open(out_file.replace('.pkl','.summary'),'w') as f: f.write("AUC {:.3f}\t Sens. {:.3f}\tSpec. {:.3f}\tSens. {:.3f}\tSpec. {:.3f}\n".format(AUC,sensitivities[0],specificities[0],sensitivities[1],specificities[1]))
#     return scores

def aeloss(net_out,ref):
    mse=nn.MSELoss(reduction='mean')
    loss_mse = 0
    for x,y in zip(net_out,ref):
        loss_mse+=mse(x,y)
    return loss_mse/len(net_out)

def loss(net_out,ref,pos_weight=1):
    bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(pos_weight).to(net_out.device))
    loss_bce = bce(net_out,ref)
    return loss_bce

def CE_loss(net_out, ref):
    ce = nn.CrossEntropyLoss()
    loss_ce = ce(net_out, ref)
    return CE_loss

def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out

def loss_fn(net_out,ref,pos_weight=1):
    bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(pos_weight).to(net_out.device))
    #bce = nn.BCELoss(reduction='mean') #, pos_weight = torch.tensor(pos_weight).to(net_out.device))
    loss_bce = bce(net_out, ref)
    return loss_bce

def save_means_plot(means, epoch, loss, figsdir='./',stds=None, fs=44100, ngf=64):
    """
    means: numpy array
    """
    try :
        temp = torch.from_numpy(means)
    except:
        temp = means
    temp1,idxs = torch.sort(temp)
    temp2 = torch.sigmoid(temp1)
    temp2 = (fs/2) * temp2

    """ mel """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, ngf+2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1)) 
    mel_freqs = hz_points[1:-1]

    """ init """
    means_init = torch.rand(ngf)*(-5)+1.3 
    means_init, _ = torch.sort(means_init)
    means_init_freq = torch.sigmoid(means_init)*(fs/2)

    fig = plt.figure(figsize=(5,3.5))

    plt.scatter(np.arange(ngf), temp2.numpy(), label='Cos-Gauss centre frequencies', marker='.')

    plt.scatter(np.arange(ngf), mel_freqs, marker='.', label='Mel centre frequencies')
    plt.xlabel('filter index')
    plt.ylabel('freq (kHz)')
    plt.yticks(np.arange(0,22050,step=5000),np.arange(0,22.05,step=5))
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5)
    
    epoch = float(epoch)
    loss = float(loss)
    plt.title(f'epoch-{epoch}-loss-{loss:.3f}')
    figpath1 = figsdir + f'center-freq-epoch-{epoch}.png'
    plt.savefig(figpath1)

    plt.close()

    del temp

    if stds is not None:
        try :
            temp = torch.from_numpy(stds)
        except:
            temp = stds
        #temp1,_ = torch.sort(temp)
        temp1 = temp[idxs]
        temp2 = torch.sigmoid(temp1)
        temp2 = (fs/2) * temp2

        fig = plt.figure(figsize=(5,3.5))

        plt.scatter(np.arange(ngf), temp2.numpy(), label='Cos-Gauss variances', marker='.')

        #plt.scatter(np.arange(ngf), mel_freqs, marker='.', label='Mel centre frequencies')
        plt.xlabel('filter index')
        #plt.ylabel('freq (kHz)')
        plt.yticks(np.arange(0,22050,step=5000),np.arange(0,22.05,step=5))
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.5)

        epoch = float(epoch)
        loss = float(loss)
        plt.title(f'epoch-{epoch}-loss-{loss:.3f}')
        figpath1 = figsdir + f'variances-epoch-{epoch}.png'
        plt.savefig(figpath1)

        plt.close()
        


def generate_examples(inputs, labels, inlens, args):
    """ inputs: B,max_len-8,3ngf
        labels: B
    """

    frames = []
    batch_size, max_len = inputs.shape[0], inputs.shape[1]
    for i in range(batch_size):
        if inlens[i] < max_len:
            frames.extend(inputs[i][:inlens[i]])
        else:
            frames.extend(inputs[i][:max_len])
    inputs = torch.stack(frames)  #(*, 3*ngf)

    labels_all = []
    for i in range(batch_size):
        if inlens[i] < max_len:
            labels_all.extend([labels[i]]*inlens[i])
        else:
            labels_all.extend([labels[i]]*max_len)

    labels_all = torch.stack(labels_all).reshape(-1,1) # (*, 1)
    assert len(inputs) == len(labels_all)
    inputs = torch.cat((inputs, labels_all), dim=1)
   
    # SHUFFLE BEFORE SUB-SAMPLE
    shuf_idx = torch.randperm(len(inputs)) 
    inputs = inputs[shuf_idx] 

    subsampling_factor = args.get('subsampling_factor', None)

    #if args['subsampling_factor'] !=0:
    #    inputs = inputs[::args['subsampling_factor']]
    if subsampling_factor is not None:
        inputs = inputs[::subsampling_factor]

    if args['oversampling']:
        oversampling_factor = args.get('oversampling_factor',-1)
    else:
        oversampling_factor = None

    """  Moved inside dataset 
    if oversampling_factor is not None:
        ind = (inputs[:,-1] == 1)
        pos_samples = inputs[inputs[:,-1]==1]
        for i in range(oversampling_factor):
            #inputs = torch.cat((inputs, inputs[ind]), dim=0)
            inputs = torch.cat((inputs, pos_samples), dim=0) 
    """

    #shuf_idx = torch.randperm(len(inputs))
    #inputs = inputs[shuf_idx]
    #labels_all = labels_all[shuf_idx]

    return inputs 

def plot_fig(y,y2=None, 
                dir='./',title='new_plot',
                leg1='leg1',leg2='leg2',
                save=True, filename='plot.png'):

    fig = plt.figure()    
    plt.plot(np.array(y), label=leg1)
    
    if y2 is not None:
        plt.plot(np.array(y2), label=leg2)
    plt.title(title)
    plt.legend()

    if save:
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(dir + '/'+ filename)
        plt.close()  

def save_checkpoint(loss, auc, epoch, args, net, 
                acfb, model, optimizer, scheduler, name=None):
    """ Save checkpoint"""
    if args.use_relWt:
        state={
            'model_state_dict': net.state_dict(),
            'acfb_state_dict': acfb.state_dict(),
            'relev_state_dict': model.relWt1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'auc': auc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state(),
            'scheduler_state_dict': scheduler.state_dict()

        }
    else:
        state={
        'model_state_dict': net.state_dict(),
        'acfb_state_dict': acfb.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'auc': auc, 
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    
        
    if args.acfb_preTrained or args.random_net:
        chk = 'ClsCheckpoint'
    else: 
        chk = 'checkpoint'

    if not os.path.isdir(args.result_folder + chk):
        os.makedirs(args.result_folder + chk)

    if name is None:
        model_path = args.result_folder + chk + '/model_epoch_' + str(epoch) +'.pth'
    else:
        model_path = args.result_folder + chk + '/' + name + '_model' + '.pth'
    print('Saving model...: ' + model_path)
    
    torch.save(state, model_path)


term_width = 10
term_width = int(term_width)

TOTAL_BAR_LENGTH = 5.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
