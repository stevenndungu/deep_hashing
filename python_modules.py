# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:25:28 2023

@author: P307791
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from timeit import time
from pathlib import Path
import os, sys, random, ast, json, glob, torch

from fastai import *
from fastai.data.all import *
from fastai.data.transforms import get_image_files
from fastai.vision.all import *

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses,  reducers
from pytorch_metric_learning.distances import lp_distance
