import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import wandb
from tqdm import tqdm
from scipy.stats import ortho_group
from copy import deepcopy
from functorch import make_functional, vmap, jacrev
# from torch.func import jacrev, functional_call

import sys
from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)