# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:25:28 2023

@author: P307791
"""
import glob
import os, random, string
import torch
import torch.nn as nn
import argparse
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
os.environ['PYTHONHASHSEED'] = 'python'

from sklearn.preprocessing import label_binarize

from torch.linalg import vector_norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random
from PIL import Image

import torch
from torch.linalg import vector_norm
from sklearn.manifold import TSNE
from sklearn import preprocessing


#For Reproducibility
def reproducibility_requirements(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print("Set seed of", str(seed),"is done for Reproducibility")

reproducibility_requirements()


def generate_unique_identifier(length):
    characters = string.ascii_letters + string.digits
    identifier = ''.join(random.choice(characters) for _ in range(length))
    return identifier


def SimplifiedTopMap(rB, qB, retrievalL, queryL, topk):
  '''
    rB - binary codes of the training set - reference set,
    qB - binary codes of the query set,
    retrievalL - labels of the training set - reference set, 
    queryL - labels of the query set, and 
    topk - the number of top retrieved results to consider.

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = 100
  '''
  num_query = queryL.shape[0]
  mAP = [0] * num_query
  for i, query in enumerate(queryL):
    rel = (np.dot(query, retrievalL.transpose()) > 0)*1 # relevant train label refs.
    hamm = np.count_nonzero(qB[i] != rB, axis=1) #hamming distance
    ind = np.argsort(hamm) #  Hamming distances in ascending order.
    rel = rel[ind] #rel is reordered based on the sorted indices ind, so that it corresponds to the sorted Hamming distances.

    top_rel = rel[:topk] #contains the relevance values for the top-k retrieved results
    tsum = np.sum(top_rel) 

    #skips the iteration if there are no relevant results.
    if tsum == 0:
        continue

    pr_count = np.linspace(1, tsum, tsum) 
    tindex = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
    pr = pr_count / tindex # precision
    mAP_sub = np.mean(pr) # AP
    mAP[i] = mAP_sub 
      


  return np.round(np.mean(mAP),4) *100 #mAP


def mAP_values(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    mAP = SimplifiedTopMap(rB, qB, retrievalL, queryL, topk)
  
    return np.round(mAP,4), r_binary, train_label, q_binary, valid_label


def get_data(path):
       
   # Load the MATLAB file
   data = loadmat(path)
   df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_train = pd.concat([df0, df1, df2, df3], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)
  
  
   # Rename the columns:
   column_names = ["descrip_" + str(i) for i in range(1, 401)] + ["label_code"]
   df_train.columns = column_names
   df_test.columns = column_names

   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_code']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

   dic_labels = { 'Bent':2,
                  'Compact':3,
                     'FRI':0,
                     'FRII':1
               }

  
   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)


   return df_train, df_test



    

    