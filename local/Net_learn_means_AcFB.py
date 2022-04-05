import torch
import torch.nn as nn
import scipy.signal
import numpy as np
import torch.nn.functional as F
import torchaudio
import random
#from utils import save_means_plot
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from matplotlib import pyplot as plt 
from pdb import set_trace as bp



def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_k(x,alpha=1):
    return 1/(1+torch.exp(-alpha * x))

def inv_sigmoid(x):
      return np.log((x+1e-8)/(1+1e-8 - x))

def mel_to_hz(mel):
    hz = (700 * (10**(mel / 2595) - 1))
    return hz

def hz_to_mel(hz):
    mel = (2595 * np.log10(1 + hz / 700))
    return mel

def init_mel(fs=44100,nfilt=64):
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(fs/2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt+2)  # Equally spaced in Mel scale   
    hz_points = mel_to_hz(mel_points) 
    mel_freqs = hz_points[1:-1]
    init_means = mel_freqs / (fs/2)
    return init_means

def init_inv_mel(fs=44100,nfilt=64, min_freq=25):
    mel_freq = init_mel(fs=fs, nfilt=nfilt)
    mel_dist = mel_freq[1:] - mel_freq[0:-1]
    mel_dist = mel_dist[::-1] # flip the mel distances
    mean_norm_freq = min_freq/(fs/2)
    invMel = [mean_norm_freq]
    for i in range(0,len(mel_dist)):
        invMel.append(invMel[i] + mel_dist[i])
    return np.array(invMel)




class Relevance_AdaptiveWeights_acousticFB1(torch.nn.Module):
    def __init__(self, ngf=80, splice = 10):
        super(Relevance_AdaptiveWeights_acousticFB1, self).__init__()
        self.ngf = ngf
        self.splice = splice
        self.patch_len = 2*self.splice + 1
        self.fc1 = nn.Linear(self.patch_len, 50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid_temp=2

    def forward(self, x):
        #x is (B, 1, ngf=80, patch_length=431)

        B,t,f = x.shape[0],x.shape[3],x.shape[2]
        
        x = F.unfold(x, (self.ngf, self.patch_len), padding=(0,self.splice)) # B, patch_len*ngf,431
        x = x.permute(0,2,1)
        x = x.reshape(B,t,f,self.patch_len) # B,431,80,21

        
        x = x.reshape(B*t*f, -1) #  (*,21)
        #x = self.sigmoid(self.fc1(x))
        x = self.tanh(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(B,t, -1)  # B, 431, 80
        # print out.shape
        #out = self.softmax(x)
        out = sigmoid_k(x,alpha=self.sigmoid_temp)

        return out

        

class channel_attention(nn.Module):
    def __init__(self,input_channel = 10, ratio=8):
        super(channel_attention,self).__init__()
        
        # input's 2nd axis is channel axis
        #self.input_channel = input_channel
        self.shared_layer_one = nn.Linear(input_channel,input_channel//ratio,
                                         bias=True)
        self.relu = nn.ReLU()
        
        self.shared_layer_two = nn.Linear(input_channel//ratio,input_channel,
                                         bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        
        #x -> N,C,H,W
        batch_size = x.size()[0]
        input_channel = x.size()[1]
        y = F.avg_pool2d(x,kernel_size=x.size()[2:]) # y-> N,C,1,1
        y = y.reshape(batch_size,-1) 
        y = self.shared_layer_one(y) # y->N,C//ratio
        y = self.relu(y)
        y = self.shared_layer_two(y)
        y = self.relu(y)
        assert y.size()[-1] == input_channel
        y = y.reshape(batch_size,1,1,input_channel)
        
        #print(x.shape)
        z = F.max_pool2d(x,kernel_size=x.size()[2:])
        z = z.reshape(batch_size,-1)
        assert z.size()[-1] == input_channel
        z = self.shared_layer_one(z)
        z = self.relu(z)
        z = self.shared_layer_two(z)
        #print(z.shape)
        assert z.shape[-1] == input_channel
        z = z.reshape(batch_size,1,1,input_channel)
        
        cbam_feature = torch.add(y,z)
        cbam_feature = self.sigmoid(cbam_feature) # batch_size,1,,1,C
        cbam_feature = cbam_feature.permute(0,3,1,2) #batch_size,C,1,1
        
        #weighted = cbam_feature * x  # batch_size,C,H,W
        
        return cbam_feature * x


class Relevance_BandWeights_acousticFB(torch.nn.Module):
    def __init__(self, n_bins=64, segment_length=51):
        super(Relevance_BandWeights_acousticFB, self).__init__()
        self.ngf = n_bins
        self.patch_length = segment_length
        self.fc1 = nn.Linear(self.patch_length, 50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x is (B, 1, patch_len=t=51, ngf=f=64)
        batch_size = x.shape[0]
        x = x.permute(0,3,2,1)   # B,80,101,1
        x = x.reshape(batch_size * x.shape[1], -1) # Bx80, 101
        x = self.sigmoid(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(batch_size, -1)  # B, 80
        # print out.shape
        out = self.softmax(x) 
        return out

class AcFB(nn.Module):
    def __init__(self, ngf=64, init=None, win_len=1102,hop_len=441, 
                    filt_h=705, patch_len=21, fs=44100, seed=42):
        super(AcFB, self).__init__()
        """ returns the learned spectrograms"""
        
        self.ngf = ngf
        self.filt_h = filt_h #int(16 * 44.1)     # = 705 # 16ms -->8ms
        if self.filt_h % 2 == 0:
            self.filt_h += 1
        self.padding = int(self.filt_h//2) #int((16 * 44.1)//2) # + 1
        self.win_length = win_len #2048 
        self.patch_length = patch_len
        self.init = init
        self.fs = fs
        self.seed = seed

        ## SEE if hamming windows are required
        
        self.len_after_conv = self.win_length-self.filt_h +1 #1344   #1102  #=win_length?
        self.hamming_window = torch.from_numpy(scipy.signal.hamming(self.win_length)).float() #.cuda()
       
        #bp()
        # MEANS INITIALIZATION
        torch.manual_seed(self.seed)
        if self.init in ['mel', 'Mel']:
            self.means = torch.nn.Parameter(torch.from_numpy(inv_sigmoid(init_mel(fs=self.fs, nfilt=self.ngf))).float()) #.cuda())
            
        elif self.init in ['inv_mel', 'invMel']:
            self.means = torch.nn.Parameter(torch.from_numpy(inv_sigmoid(init_inv_mel(fs=self.fs, nfilt=self.ngf))).float()) #.cuda())
            

        elif self.init in ['rand1', 'Rand1']:
            self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-5) + 1.6).float()) #.cuda())

        elif self.init in ['linear', 'Linear']:
            lin = torch.linspace(0.0, 1.0, steps=self.ngf+2)
            self.means = torch.nn.Parameter(inv_sigmoid(lin[1:-1]).float())


        elif self.init in ['rand2', 'Rand2']:
            self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-6.5) + 1.2).float()) #.cuda())

        else:
            raise ValueError("AcFB initialization not implemented")
        print(f'Initializing acfb with {self.init} centre frequencies')

        t = range(-self.filt_h//2, self.filt_h//2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float() +1 #.cuda() + 1
        
        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_conv, 1), stride=(1,1))

    def get_kernels(self):
        means_sorted = torch.sort(self.means)[0]
        kernels=torch.zeros([self.ngf, self.filt_h])
        for i in range(self.ngf):
            kernels[i, :] = (torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * 
                            torch.exp(-(((self.temp)**2)/(2*(((1/torch.sigmoid(means_sorted[i]+1e-3))*10)**2+1e-5)))))

        return kernels


    def forward(self, x):
        # x = B,C,H,W = B,C,win_len,t
        self.hamming_window = self.hamming_window.to(x.device) # check device
        #self.means.to(x.device)
        #self.temp = self.temp.cuda(x.get_device())
        self.temp = self.temp.to(x.device)

        patch_length = x.shape[3] # no of time frames
        #assert patch_length == 431
        means_sorted = torch.sort(self.means)[0].to(x.device)
        kernels = torch.zeros([self.ngf, self.filt_h]).to(x.device) #.cuda()

        for i in range(self.ngf):
            kernels[i, :] = (torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * 
                            torch.exp(-(((self.temp)**2)/(2*(((1/torch.sigmoid(means_sorted[i]+1e-3))*30)**2+1e-5)))))
        kernels = torch.reshape(kernels,(kernels.shape[0],1,kernels.shape[1], 1)) # ngf, 1, filt_h, 1

        x = x.permute(0,1,3,2) # B,1,431,2048=B,1,W,H check
        
        # check if Hamming window should be used
        x = x * self.hamming_window
        x = x.permute(0,2,3,1) # B, W, H,C
        
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size * x.shape[1],1,self.win_length,1)) # B*W, 1, H, 1
        #------------- padding ? --------------------------
        x = F.conv2d(x, kernels) #, padding = (self.padding, 0)) 

        # here x = B*w, ngf, H, 1 ; w = patch_length, H=len_after_conv=win_length
        x = torch.reshape(x, (batch_size, patch_length, self.ngf, self.len_after_conv)).permute(0,2,3,1)  # B, ngf, H, W
        x = torch.abs(x)**2
        #torch.abs_(x)**2
        #x = self.avg_pool_layer(x) + 1e-3
        #x = torch.log(self.avg_pool_layer(x) + 1e-3)
        x = self.avg_pool_layer(x)
        # epsilon to be added?------------------------
        x = x.permute(0,2,1,3) # B, 1, ngf, W 
        x = torchaudio.functional.amplitude_to_DB(x, multiplier=10,db_multiplier=0,amin=1e-10) #(x,mult,amin,db_multiplier,top_db)
        
        assert x.shape[2] == self.ngf
        assert x.shape[3] == patch_length
        return x

class AcFB_preTrained(nn.Module):
    def __init__(self, ngf=64, init=None, win_len=1102,hop_len=441, 
                    filt_h=705, patch_len=21, fs=44100,seed=42):
        super(AcFB_preTrained, self).__init__()
        """ returns the learned spectrograms"""
        
        self.ngf = ngf
        self.filt_h = filt_h #int(16 * 44.1)     # = 705 # 16ms -->8ms
        if self.filt_h % 2 == 0:
            self.filt_h += 1
        self.padding = int(self.filt_h//2) #int((16 * 44.1)//2) # + 1
        self.win_length = win_len #2048 
        self.patch_length = patch_len
        self.init = init
        self.fs = fs
        self.seed=seed

        ## SEE if hamming windows are required
        
        self.len_after_conv = self.win_length-self.filt_h +1 #1344   #1102  #=win_length?
        self.hamming_window = torch.from_numpy(scipy.signal.hamming(self.win_length)).float() #.cuda()
       
        torch.manual_seed(self.seed)
        # MEANS INITIALIZATION
        if self.init in ['mel', 'Mel']:
            self.means = torch.nn.Parameter(torch.from_numpy(init_mel(fs=self.fs, nfilt=self.ngf)).float()) #.cuda())
            print('Initializing acfb with {} centre frequencies')
        else:
            self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-5) + 1.3).float()) #.cuda())
        
        print(f'Initializing acfb with {self.init} centre frequencies')

        t = range(-self.filt_h//2, self.filt_h//2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float() +1 #.cuda() + 1
        
        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_conv, 1), stride=(1,1))

    def forward(self, x):
        # x = B,C,H,W = B,C,win_len,t
        self.hamming_window = self.hamming_window.to(x.device) # check device
        #self.means.to(x.device)
        #self.temp = self.temp.cuda(x.get_device())
        self.temp = self.temp.to(x.device)

        patch_length = x.shape[3] # no of time frames
        #assert patch_length == 431
        means_sorted = torch.sort(self.means)[0].to(x.device)
        kernels = torch.zeros([self.ngf, self.filt_h]).to(x.device) #.cuda()

        for i in range(self.ngf):
            kernels[i, :] = (torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * 
                            torch.exp(-(((self.temp)**2)/(2*(((1/torch.sigmoid(means_sorted[i]+1e-3))*10)**2+1e-5)))))
        kernels = torch.reshape(kernels,(kernels.shape[0],1,kernels.shape[1], 1)) # ngf, 1, filt_h, 1

        x = x.permute(0,1,3,2) # B,1,431,2048=B,1,W,H check
        
        # check if Hamming window should be used
        x = x * self.hamming_window
        x = x.permute(0,2,3,1) # B, W, H,C
        
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size * x.shape[1],1,self.win_length,1)) # B*W, 1, H, 1
        #------------- padding ? --------------------------
        x = F.conv2d(x, kernels) #, padding = (self.padding, 0)) 

        # here x = B*w, ngf, H, 1 ; w = patch_length, H=len_after_conv=win_length
        x = torch.reshape(x, (batch_size, patch_length, self.ngf, self.len_after_conv)).permute(0,2,3,1)  # B, ngf, H, W
        x = torch.abs(x)**2
        #torch.abs_(x)**2
        #x = self.avg_pool_layer(x) + 1e-3
        #x = torch.log(self.avg_pool_layer(x) + 1e-3)
        x = self.avg_pool_layer(x)
        # epsilon to be added?------------------------
        x = x.permute(0,2,1,3) # B, 1, ngf, W 
        x = torchaudio.functional.amplitude_to_DB(x, multiplier=10,db_multiplier=0,amin=1e-5,top_db=80.0) #(x,mult,amin,db_multiplier,top_db)
        
        assert x.shape[2] == self.ngf
        assert x.shape[3] == patch_length
        return x

class AcFB_FreeMeanVar(nn.Module):
    def __init__(self, ngf=64, init=None, win_len=1102,hop_len=441, 
                    filt_h=705, patch_len=21, fs=44100, seed=42, plot_kernels=False):
        super(AcFB_FreeMeanVar, self).__init__()
        """ returns the learned spectrograms"""
        
        self.ngf = ngf
        self.filt_h = filt_h #int(16 * 44.1)     # = 705 # 16ms -->8ms
        if self.filt_h % 2 == 0:
            self.filt_h += 1
        self.padding = int(self.filt_h//2) #int((16 * 44.1)//2) # + 1
        self.win_length = win_len #2048 
        self.patch_length = patch_len
        self.init = init
        self.fs = fs
        self.plot_kernels = plot_kernels
        self.seed=seed

        ## SEE if hamming windows are required
        
        self.len_after_conv = self.win_length-self.filt_h +1 #1344   #1102  #=win_length?
        self.hamming_window = torch.from_numpy(scipy.signal.hamming(self.win_length)).float() #.cuda()
       
        torch.manual_seed(self.seed)
        # MEANS INITIALIZATION
        if self.init in ['mel', 'Mel']:
            self.means = torch.nn.Parameter(torch.from_numpy(init_mel(fs=self.fs, nfilt=self.ngf)).float()) #.cuda())
        else:
            self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-5) + 1.3).float().sort()[0]) #.cuda())
        
        # STD INITIALIZATION
        self.stds = torch.nn.Parameter((self.means))

        t = range(-self.filt_h//2, self.filt_h//2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float() +1 #.cuda() + 1
        
        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_conv, 1), stride=(1,1))

    def forward(self, x):
        # x = B,C,H,W = B,C,win_len,t
        self.hamming_window = self.hamming_window.to(x.device) # check device
        #self.means.to(x.device)
        #self.temp = self.temp.cuda(x.get_device())
        self.temp = self.temp.to(x.device)

        patch_length = x.shape[3] # no of time frames
        #assert patch_length == 431
        means_sorted, idx = torch.sort(self.means)
        means_sorted = means_sorted.to(x.device)
        stds_ordered = self.stds[idx].to(x.device)

        kernels = torch.zeros([self.ngf, self.filt_h]).to(x.device) #.cuda()

        for i in range(self.ngf):
            kernels[i, :] = (torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * 
                            torch.exp(-(((self.temp)**2)/(2*4*(((1/torch.sigmoid(stds_ordered[i]))*3)**2+1e-5))))) # denom factor 10-->4 to make it exact factor
        
        """
        if self.plot_kernels:
            kk = kernels.detach().cpu().numpy()
            kk1 = np.fft.fft(kk, axis=-1)
            kk1 = np.abs(kk1[:,:self.filt_h//2+1])
            for i in range(self.ngf):
                plt.plot(kk1[i,:])
            plt.savefig('kernels_plot_vars.png')
            plt.close()

        """

        kernels = torch.reshape(kernels,(kernels.shape[0],1,kernels.shape[1], 1)) # ngf, 1, filt_h, 1

        x = x.permute(0,1,3,2) # B,1,431,2048=B,1,W,H check
        
        # check if Hamming window should be used
        x = x * self.hamming_window
        x = x.permute(0,2,3,1) # B, W, H,C
        
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size * x.shape[1],1,self.win_length,1)) # B*W, 1, H, 1
        #------------- padding ? --------------------------
        x = F.conv2d(x, kernels) #, padding = (self.padding, 0)) 

        # here x = B*w, ngf, H, 1 ; w = patch_length, H=len_after_conv=win_length
        x = torch.reshape(x, (batch_size, patch_length, self.ngf, self.len_after_conv)).permute(0,2,3,1)  # B, ngf, H, W
        x = torch.abs(x)**2
        #torch.abs_(x)**2
        #x = self.avg_pool_layer(x) + 1e-3
        #x = torch.log(self.avg_pool_layer(x) + 1e-3)
        x = self.avg_pool_layer(x)
        # epsilon to be added?------------------------
        x = x.permute(0,2,1,3) # B, 1, ngf, W 
        x = torchaudio.functional.amplitude_to_DB(x, multiplier=10,db_multiplier=0,amin=1e-5,top_db=80.0) #(x,mult,amin,db_multiplier,top_db)
        
        assert x.shape[2] == self.ngf
        assert x.shape[3] == patch_length
        return x


