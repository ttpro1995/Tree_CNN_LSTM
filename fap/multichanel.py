import torch
import torch.nn as nn
from torch.autograd import Variable as Var

padding = (5-1)/2
emb_size = 300
conv = nn.Conv2d(1, 200, (5, emb_size), padding=(padding,0))

sentence = Var(torch.rand(24, 1, emb_size))
sentence2 = Var(torch.rand(24, 1, emb_size))