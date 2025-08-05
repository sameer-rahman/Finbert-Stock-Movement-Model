# notebook_utils.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import lightgbm as lgb
import shap
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Get CUDA device if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    return device


def set_matplotlib_style():
    """Set default matplotlib style."""
    plt.style.use("seaborn-vcolor")
    plt.rcParams["figure.figsize"] = (10, 5)
