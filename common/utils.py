import torch
import numpy as np
import os
import random

# 保证训练过程可复现，使用确定的随机数种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # CuDNN操作确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False