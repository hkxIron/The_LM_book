import torch
import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True # 会显著降低速度，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法，个人理解，如果没有用到drop out 等模型内部的随机trick，这一项可以去除