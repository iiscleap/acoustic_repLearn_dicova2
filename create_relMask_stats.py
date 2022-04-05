import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from local.Net_learn_means_AcFB import AcFB, Relevance_AdaptiveWeights_acousticFB1, Relevance_BandWeights_acousticFB
from local.models import LSTMClassifier
from local.utils_funcs import *
import argparse, configparser
import scipy.io as sio

args={

    'num_freq_bin': 64,
    'model_config':'conf/model_config',
    'use_relWt':True,
    'relevance_type':'adaptiveWt',
    'relContext':51,
    'mean_var_norm': True,
    'use_mixup':False,
    'acfb_preTrained':True,
    'use_skipConn':False,
    'deltas':True,
    'feat_config':'conf/feature.conf',
    "fs":44100
}

audiocatgory='breathing'
outerid=0
#resdir='breathing/feats_results_BLSTM_segment_PRETRAIN_FB_CAMBRIDGE_SR44_WD0-0_BSZ8_finetune_sigFac2_acfb0.001_cls0.0001_lambda0.001_tanh_initrand1_sch/'
resdir='breathing/feats_results_BLSTM_segment_PRETRAIN_FB_CAMBRIDGE_SR44_WD0-0_BSZ8_finetune_sigFac2_relHiddenTanh_acfb0.001_cls0.0001_lambda0.001_tanh_initrand1_sch'
resdir=resdir+"/"+str(outerid)
chkdir=resdir+'/ClsCheckpoint'
model_path=chkdir+'/best_frame_auc_model.pth'
max_count=100


model_args = {'input_dimension':3*64}
if args['model_config'] != 'None':
    temp = configparser.ConfigParser()
    temp.read(args['model_config'])
    for key in temp['default'].keys():
        model_args[key]=convertType(temp['default'][key])
print(model_args)



class feature_extractor_v2():
    def __init__(self,args):

        self.args=args	
        self.vad_threshold = 1e-4 #args.get('threshold',1e-4)
        self.resampling_rate = 44100 #int(self.args['resampling_rate'])
        if self.args['feature_type'] == 'MFCC':
            self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate, 
                                                                n_mfcc=int(self.args['n_mfcc']), 
                                                                melkwargs={
                                                                    'n_fft': int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
                                                                    'n_mels': int(self.args['n_mels']), 
                                                                    'f_max': int(self.args['f_max']), 
                                                                    'hop_length': int(float(self.args['hop_length'])*1e-3*self.resampling_rate)})
        elif self.args['feature_type'] in ['MelSpec','logMelSpec'] :
            self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate= 44100 ,#self.resampling_rate,
                                                                    n_fft= 1024, #int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
                                                                    n_mels= 64,#int(self.args['n_mels']), 
                                                                    f_max= 22050, #int(self.args['f_max']), 
                                                                    hop_length= 441 #int(float(self.args['hop_length'])*1e-3*self.resampling_rate)
                                                                    )
        else:
            raise ValueError('Feature type not implemented')

    def extract(self,filepath):

        s,fs = torchaudio.load(filepath)
        s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(self.resampling_rate)]])           
        s = s/torch.max(torch.abs(s))
        #sad = compute_SAD(s.numpy(),self.resampling_rate,threshold=self.vad_threshold)
        #s = s[np.where(sad==1)]     
        F = self.feature_transform(s)
        if self.args['feature_type'] == 'logMelSpec': 
            F = torchaudio.functional.amplitude_to_DB(F,multiplier=10,amin=1e-10,db_multiplier=0)
        return F.T



class getJointNet(torch.nn.Module):
    def __init__(self, acfb, net, args):
        super(getJointNet, self).__init__()

        self.args = args
        self.acfb = acfb
        self.context_len = args['relContext']
                
        if self.args['use_relWt']:
            if self.args['relevance_type'] == 'bandWt':
                self.relWt1 = Relevance_BandWeights_acousticFB(n_bins=len(self.acfb.means), segment_length=51)
            elif self.args['relevance_type'] == 'adaptiveWt':
                self.relWt1 = Relevance_AdaptiveWeights_acousticFB1(ngf=len(self.acfb.means), splice = self.context_len)
        self.classifier = net

    def get_tf(self, inputs):
        inputs = inputs[0]
        inputs = inputs.unsqueeze(1).unsqueeze(1).permute(0,1,3,2)
        inputs=self.acfb(inputs)
        if self.args['use_relWt']:
            if self.args['relevance_type'] == 'bandWt':
                #inputs = inputs.permute(0,1,3,2) # B,1,t,f
                relWt = self.relWt1(inputs)
            elif self.args['relevance_type'] == 'adaptiveWt':
                relMask = self.relWt1(inputs) # B,t,f
            return inputs, relMask
            
        return inputs
    




    def forward(self,inputs, labels, mode='val'):
        inlens = [x.shape[0] for x in inputs]
        inputs = pad_sequence(inputs, batch_first=True)
        # Make shape B,C,H,W
        inputs = inputs.unsqueeze(1).permute(0,1,3,2)
        
        if self.args['acfb_preTrained']:
            with torch.no_grad():
                inputs = self.acfb(inputs)
                #return inputs #================
        else:
            inputs = self.acfb(inputs)

            #return inputs  #=============
        #return inputs
        
        #inputs: B,1,f,t
        if self.args['use_relWt']: 
            if self.args['relevance_type'] == 'bandWt':
                inputs = inputs.permute(0,1,3,2) # B,1,t,f
                relWt = self.relWt1(inputs)
                relWt = relWt.unsqueeze(1).unsqueeze(1)
                if self.args['use_skipConn']:
                    inputs = inputs * relWt + inputs
                else:
                    inputs = inputs * relWt
                inputs1 = inputs.permute(0,1,3,2).contiguous() # B,1,f,t

            elif self.args['relevance_type'] == 'adaptiveWt':
                relMask = self.relWt1(inputs) # B,t,f
                relMask = relMask.unsqueeze(1).permute(0,1,3,2).contiguous() # B,1,f,t
                assert inputs.shape == relMask.shape
                if self.args['use_skipConn']:
                    inputs = inputs * relMask + inputs # B,1,f,t
                else:
                    inputs = inputs * relMask

        # inputs: B,1,f,t
        if self.args['deltas']:

            inputs_deltas = torchaudio.functional.compute_deltas(inputs, win_length=9) # mode='reflect'
            inputs_deltas_deltas = torchaudio.functional.compute_deltas(inputs_deltas, win_length=9)

            #inputs =  torch.cat((inputs[:,:,4:-4,:],inputs_deltas[:,:,2:-2,:],inputs_deltas_deltas),1) # B,3ngf,T-8,1
            inputs =  torch.cat((inputs, inputs_deltas, inputs_deltas_deltas),2) # B,1,f,t


        # MVN OVER THE CONCATED FEATURES
        #bp()
        # Mixup
        self.labels_a, self.labels_b = None, None
        if self.args['use_mixup'] and mode != 'val':
           inputs, self.labels_a, self.labels_b, self.lam = mixup_data(inputs, labels, inputs.device, alpha=0.4) #only mixup

        # inputs: B,1,f,t
        if self.args['mean_var_norm']:
            inputs = inputs - torch.mean(inputs, dim=-1,keepdim=True)
        #if self.args['apply_var_norm']:
            inputs = inputs / (torch.std(inputs, dim=-1,keepdim=True) + 1e-8)
        inputs = inputs.squeeze(1).permute(0,2,1).contiguous()
        x = [inputs[i,:inlens[i]] for i in range(inputs.shape[0])] 
        x, targets = self.classifier(x, labels)
        
        if self.args['use_mixup'] and mode != 'val':
            return x, torch.stack(self.labels_a), torch.stack(self.labels_b), self.lam
        return x, targets




net = LSTMClassifier(model_args)

acfb = AcFB(ngf=args['num_freq_bin'], init='rand1',
    win_len = 1102,
    hop_len = 441,
    filt_h = 353,
    seed = 42)

model = getJointNet(acfb, net, args)

checkpoint=torch.load(model_path,map_location='cpu')
model.acfb.load_state_dict(checkpoint['acfb_state_dict'])
model.classifier.load_state_dict(checkpoint['model_state_dict'])
if args['use_relWt']:
    model.relWt1.load_state_dict(checkpoint['relev_state_dict'])
print(model)


feat_args={
    'threshold':1e-4,
    'feature_type':'logMelSpec'

}

p_cnt=0
n_cnt=0


device=torch.device('cpu')

feat_config = configparser.ConfigParser()
feat_config.read(args['feat_config'])


FE = FrameFeature_extractor(feat_config) ## Doesn't perform VAD
FE_mel=feature_extractor_v2(feat_args)

# Loop over all files
file_list='data/{}/train'.format(outerid)
file_list = open(file_list).readlines()
file_list = [line.strip() for line in file_list]

np.random.seed(42)
np.random.shuffle(file_list)

fids=[line.split(" ")[0] for line in file_list]

n_pos=0
for line in file_list:
    id,label=line.split(" ")
    if label=='p':
        n_pos+=1

# if n_pos > max_count:
#     max_count=n_pos 
#     print("setting max_count={}".format(max_count))

sub_list=[]

for line in file_list:
    id, label=line.split(" ")
    if label == 'p':
        if p_cnt < max_count:
            sub_list.append(line)
            p_cnt+=1

    if label == 'n':
        if n_cnt < max_count:
            sub_list.append(line)
            n_cnt+=1
    if n_cnt == max_count and p_cnt==max_count:
        break



p_cnt=0
n_cnt=0

feats_file = 'data/breathing_vad.scp'
feats_lines=open(feats_file,'r').readlines()

paths={}
for line in feats_lines:
    key, path = line.strip().split(" ")
    paths[key]=path


scores={}
pos_mask=[]
neg_mask=[]

for line in tqdm.tqdm(sub_list):
    
    fileId, label=line.split(" ")
    # Prepare features
    path = paths[fileId]
    #print(fileId, path)
    F = FE.extract(path)	# F should be data frames: 
    F = torch.from_numpy(F)
    F = F.float()
    feat = F.to(device)

    # Input mode
    seg_mode = 'file' #config['training_dataset'].get('mode','file')
    if seg_mode=='file':
        feat = [feat]
    elif seg_mode=='segment':
        segment_length = 51 #int(config['training_dataset'].get('segment_length',300))
        segment_hop = 10 #int(config['training_dataset'].get('segment_hop',10))
        feat = [feat[i:i+segment_length,:] for i in range(0,max(1,F.shape[0]-segment_length),segment_hop)]
    else:
        raise ValueError('Unknown eval model')

    with torch.no_grad():
        output, mask = model.get_tf(feat)
        output=output.detach().squeeze()
        mask=mask.detach().squeeze()
        #print(output.shape, mask.shape)

        if label=='p':
            pos_mask.append(mask)
            p_cnt+=1
        elif label=='n':
            neg_mask.append(mask)
            n_cnt+=1
        else:
            raise "Unknown label!"

    specdir="{}/figs/SPEC-MASKS/".format(resdir)
    if not os.path.exists(specdir):
        os.makedirs(specdir)
    
    outfile=specdir+"/spec-{}:{}.png".format(fileId,label)
    # fig = plt.figure()
    # plt.imshow(np.flipud(output.permute(1,0).numpy()),cmap='jet')
    # plt.colorbar(shrink=0.5)
    # title=outfile.split("/")[-1].split(".")[0]
    # plt.title(title)
    # plt.yticks(np.arange(0,64,step=20),np.arange(64,0,step=-20))
    # plt.savefig(outfile)
    # plt.close()
    dumpf=outfile[:-4]+".mat"
    sio.savemat(dumpf,mdict={'data':np.flipud(output.permute(1,0).numpy())})

    outMaskfile=specdir+"/Mask-{}:{}.png".format(fileId,label)
    # fig = plt.figure()
    # plt.imshow(np.flipud(mask.permute(1,0).numpy()),cmap='jet')
    # plt.colorbar(shrink=0.5)
    # title=outMaskfile.split("/")[-1].split(".")[0]
    # plt.title(title)
    # plt.yticks(np.arange(0,64,step=20),np.arange(64,0,step=-20))
    # plt.savefig(outMaskfile)
    # plt.close()
    dumpMaskf=outMaskfile[:-4]+".mat"
    sio.savemat(dumpMaskf,mdict={'data':np.flipud(mask.permute(1,0).numpy())})
    print("dumped:{}".format(dumpMaskf))
#         #break
pos_com_mask=torch.cat(pos_mask,dim=0)
neg_com_mask=torch.cat(neg_mask,dim=0)


print(pos_com_mask.shape, neg_com_mask.shape)
pos_com_mask=pos_com_mask.permute(1,0)
neg_com_mask=neg_com_mask.permute(1,0)

sys.exit(0)

#%%

import scipy
import scipy.stats
import scipy.io as sio

def get_stats(arr):
    stats={}
    arr=arr.reshape(-1)
    stats['mean']=np.mean(arr)
    stats['std']=np.std(arr)
    stats['kurt']=scipy.stats.kurtosis(arr,axis=-1,fisher=False)
    stats['skew']=scipy.stats.skew(arr,axis=-1)

    return stats

n_blocks=10
figdir="{}/figs/clsRelWt/{}".format(resdir,n_blocks)
if not os.path.exists(figdir):
    os.makedirs(figdir)


''' relevance profile across freqs'''
pos_freq_dist=torch.mean(pos_com_mask,dim=1)
neg_freq_dist=torch.mean(neg_com_mask,dim=1)

distdir="{}/figs/clsRelWt/".format(resdir)
sio.savemat(distdir+"/pos_wt_dist.mat",mdict={'data':pos_freq_dist.numpy()})
sio.savemat(distdir+"/neg_wt_dist.mat",mdict={'data':neg_freq_dist.numpy()})



bw=args['num_freq_bin']//n_blocks
start=0

#for i in range(n_blocks):
while start < args['num_freq_bin']:
   
    end=start+bw
    pos_band_weights=pos_com_mask[start:end,:]
    neg_band_weights=neg_com_mask[start:end,:]

    pos_band_weights=pos_band_weights.numpy().reshape(-1)
    neg_band_weights=neg_band_weights.numpy().reshape(-1)
    pos_stats=get_stats(pos_band_weights)
    neg_stats=get_stats(neg_band_weights)

    # fig,ax=plt.subplots(1,2, tight_layout=True) #,figsize=(6,4),
    # ax[0].hist(pos_band_weights)
    # ax[1].hist(neg_band_weights)
    # ax[0].set_title("{}-{}:m:{:.3f},std:{:.3f},ku:{:.2f},sk:{:.2f}".format(start,end,pos_stats['mean'],pos_stats['std'],pos_stats['kurt'],pos_stats['skew']),fontsize='small')
    # ax[1].set_title("{}-{}:m:{:.3f},std:{:.3f},ku:{:.2f},sk:{:.2f}".format(start,end,neg_stats['mean'],neg_stats['std'],neg_stats['kurt'],neg_stats['skew']),fontsize='small')
    # #ax[1].set_title("{}th block neg dist".format(i))
    # plt.savefig("{}/bands:{}-{}.png".format(figdir,start,end))
    # #plt.show()
    # plt.tight_layout()
    # plt.close(fig)

    #pos_stats=get_stats(pos_band_weights)
    #neg_stats=get_stats(neg_band_weights)

    #print("bands:{}-{}, pos_mean={:.4f}, neg_mean={:.4f},".format(start, end ,pos_mean.item(),neg_mean.item()))
    print("*"*20+" {}-{} bands ".format(start,end)+"*"*20)
    print("pos::")
    for key in pos_stats.keys():
        print("{}:{:.4f}".format(key,pos_stats[key]))

    print("neg:")
    for key in neg_stats.keys():
        print("{}:{:.4f}".format(key,neg_stats[key]))
    print("\n")
    
    

    start=end
    #break
