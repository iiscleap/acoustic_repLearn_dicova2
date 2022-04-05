#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:32:50 2021

@author: srikanthr
"""

import os,sys
from scoring import *
from utils_funcs import *

def ArithemeticMeanFusion(scores,weight=None):
	'''
	Artihmetic mean fusion of scores
	scores: is a list of dictionaries with scores as key,value pairs
	weights: list of weights
	'''
	if weight==None:
		weight=[1/len(scores) for i in range(len(scores))]
	assert len(weight)==len(scores)

	if len(scores)==1: 
		# Nothing to fuse
		return scores
	else:
		keys = set(scores[0].keys())
		# get common participants
		for i in range(1,len(scores)):
			if keys != set(scores[i].keys()): 
				print("WARNING: Expected all scores to come from same set of participants")
				keys = keys.intersection(set(scores[i].keys()))
		# do weighted sum for each participant
		fused_scores={}
		for key in keys:
			s = [weight[i]*scores[i][key] for i in range(len(scores))]
			fused_scores[key]=sum(s)/sum(weight)
		return fused_scores
		

def do_score_fusion(ffolders,outscores,normalize=False,weights=None):
#%%
    ''' Function to do score fusion
    Normalizes the scores if normalize is set to True
    The scores are averaged and written to outscores
    '''
    scores={}
    for i in range(len(ffolders)):
    	temp=to_dict(ffolders[i])
    	if normalize:
    		# Map the scores between 0-1
    		min_score = temp[min(temp,key=temp.get)]
    		max_score = temp[max(temp,key=temp.get)]
    		score_range = max_score-min_score
    		for item in temp:
    			temp[item] = (temp[item]-min_score)/score_range
    	scores[i] = temp
    # Fused score is an average of the test scores
    fused_scores=ArithemeticMeanFusion([scores[i] for i in range(len(ffolders)) ],weights)
    with open(outscores,'w') as f:
    	for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))    

#%%
normalize = False if sys.argv[-1]=='False' else True
outscores=sys.argv[-2]
ffolders = [sys.argv[i] for i in range(1,len(sys.argv)-2)]
R = do_score_fusion(ffolders,outscores,normalize=normalize)
