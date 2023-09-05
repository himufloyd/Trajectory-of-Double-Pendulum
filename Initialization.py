from google.colab import drive
drive.mount('/content/drive', force_remount = True)

%%capture
# !pip install pathos
# !pip install DeepXDE==0.11.2
# %cd /content/drive/MyDrive/Self Project/SUBHAM/DeepONet_code
# !git clone https://github.com/lululxvi/deeponet.git


import numpy as np
from scipy.optimize import root
import os
from multiprocessing import Pool
import timeit

from torch.utils.data import Dataset
from tqdm.notebook import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
