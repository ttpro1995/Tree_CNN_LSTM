import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from model import ConvModule

def test_conv():
    emb_size = 300
    sentence = Var(torch.rand(24, 1, emb_size))
    conv = ConvModule(0, emb_size, 300, 5)
