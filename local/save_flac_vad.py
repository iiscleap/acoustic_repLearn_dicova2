#import numpy as np 
#import soundfile as sf
#import librosa 
#import torchaudio
import os, sys
#import torch
import numpy as np
import torch
import torchaudio

RESAMPLING_RATE=44100
VAD_THRESHOLD=1e-4

wavdir=sys.argv[1]
#wavdir='/home/debottamd/covid-19-sounds/covid19_data_0426/audio/'
#wavdir='../audio/'
outdir=sys.argv[2]
#outdir='./aug_flac'
wavfile=sys.argv[3]
#wavfile = '../meta_files/extra_meta/Breath_full_wav.scp'
outfil=sys.argv[4]
outscp=open(outfil,'w')
#outscp=open('extra_meta/Breath_flac_wav.scp','w')

SRs = []
flist = []

if not os.path.exists(outdir):
    os.makedirs(outdir)
wavlines = open(wavfile,'r').readlines()

def compute_SAD(sig,fs,threshold=0.0001,sad_start_end_sil_length=100, sad_margin_length=50):

    sad_start_end_sil_length = int(sad_start_end_sil_length*1e-3*fs)
    sad_margin_length = int(sad_margin_length*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>threshold] = 1
    sad = np.zeros(sig.shape)
    #sad2 = np.zeros(sig.shape)

    N, idx  = sample_activity.shape[1], 0
    """
    while idx < N:
        if sample_activity[0,idx] == 1:
            sad[0,idx-sad_margin_length:idx+sad_margin_length] = 1
            idx += sad_margin_length
        else:
            idx += 1
    
    """
    for i in range(sample_activity.shape[1]):
        if sample_activity[0,i] == 1:
            sad[0,i-sad_margin_length:i+sad_margin_length] = 1
    #bp()
    
    sad[0,0:sad_start_end_sil_length] = 0
    sad[0,-sad_start_end_sil_length:] = 0
    return sad


for line in wavlines:
    uid, path = line.strip().split(" ")
    print(uid, path, os.path.exists(path))
    if wavdir!="":
        filepath = wavdir + "/" + path
    else:
        filepath=path
        
    filepath = filepath.split(".")[0]+'.wav' ## All are in .wav format ??
    """
    x, fs = sf.read(filepath)
    SRs.append(fs)
    outpath = outdir+"/"+path
    outpath  = outpath.split(".")[0] + '.flac'
    outfolder = outdir + "/" +'/'.join(path.split("/")[:-1])
    print(outfolder)
    #break
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    x = torch.from_numpy(x)
    x = x[:,1].float().reshape(-1,1)
    torchaudio.backend.sox_io_backend.save(outpath, x, 44100) # format="flac")
    #sf.write(outpath, x, 44100, format='FLAC')
    flist.append(uid + " "+ outpath)
    break
    """
    tempdir='temp_audio'
    temp_path = tempdir+"/"+path
    temp_path  = temp_path.split(".")[0] + '.flac'
    temp_folder = tempdir + "/" +'/'.join(path.split("/")[:-1])
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    os.system('ffmpeg -i '+ filepath + ' -af aformat=s16:44100 '+temp_path + ' -loglevel quiet -y')  

    s,fs = torchaudio.load(temp_path)
    s = s[0,:].reshape(1,-1) # Convert to mono
    s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(RESAMPLING_RATE)]])           
    s = s/torch.max(torch.abs(s))
    sad = compute_SAD(s.numpy(), RESAMPLING_RATE, threshold=VAD_THRESHOLD)
    s = s[np.where(sad==1)] 

    out_path = outdir+"/"+path
    out_path  = out_path.split(".")[0] + '.flac'
    out_folder = outdir + "/" +'/'.join(path.split("/")[:-1])   
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    torchaudio.save(out_path, s, RESAMPLING_RATE)
    flist.append(uid + " "+ out_path)  
    #break

    
for line in flist:
    outscp.write(line+"\n")
outscp.close()

print("Coversion to vad FLAC done! ")
    

    
    
    
    