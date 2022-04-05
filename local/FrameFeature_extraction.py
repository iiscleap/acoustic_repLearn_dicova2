import numpy as np
import librosa, pickle
import tqdm as tqdm
from utils_funcs import *
import torchaudio, torch
import argparse,configparser

#%%
def main(config,filelist,outdir):
    	
	feats_list = []

	temp = open(filelist).readlines()
	filepaths={}
	for line in temp:
		idx,path = line.strip().split()
		filepaths[idx]=path
	FE = FrameFeature_extractor(config)
	for item in tqdm.tqdm(filepaths):
		F = FE.extract(filepaths[item])
		outname = '{}/{}.pkl'.format(outdir,item)
		with open(outname,'wb') as f: pickle.dump(F,f)	

		feats_list.append((item, outname))
		#break

	with open(outdir+"/feats.scp","w") as f:
	        for file_id, out_file_name in feats_list:
    	                f.write(file_id+" "+out_file_name+"\n")

    		
    	


if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config','-c',required=True)
	parser.add_argument('--filelist','-f',required=True)
	parser.add_argument('--outdir','-o',required=True)	
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)

	main(config, args.filelist, args.outdir)
