import torch
import numpy as np
import os
import random
import re

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

# 对字符串列表按字母-数字排序
def sorted_alphanum(img_names):
    # 定义一个转换函数：字符转成小写，如果是数字提取数值
    convert = lambda str: int(str) if str.isdigit() else str.lower()
    # 定义一个排序key函数，按字母-数字切分，得到列表作为key
    alphanum = lambda str: [convert(x) for x in re.split('([0-9]+)', str)]
    return sorted(img_names, key=alphanum)