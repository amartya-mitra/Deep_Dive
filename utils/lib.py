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
try:
    from functorch import make_functional, vmap, jacrev
except ImportError:
    from torch.func import functional_call, vmap, jacrev
    from torch._functorch.eager_transforms import make_functional

import sys
from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)