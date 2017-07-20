import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from model import ConvModule, MultiConvModule

# def test_conv():
#     emb_size = 300
#     sentence = Var(torch.rand(1, 1, emb_size))
#     conv = ConvModule(0, emb_size, 200, 11)
#     output = conv(sentence)
#     print (output.size())
#
# def test_multi_conv():
#     emb_size = 300
#     sentence = Var(torch.rand(24, 1, emb_size))
#     conv = MultiConvModule(0, 300, 1, [100, 200, 300], [3, 5, 7])
#     print (conv)
#     output = conv(sentence)
#     print (output.size())
#
def test_multi_conv():
    emb_size = 300
    sentence = Var(torch.rand(24, 1, emb_size))
    conv = MultiConvModule(0, 300, 1, [100, 200, 300], [3, 5, 7])
    print (conv)
    output = conv(sentence)
    print (output.size())

def test_multi_conv_pooling():
    emb_size = 300
    sentence = Var(torch.rand(24, 1, emb_size))
    conv = MultiConvModule(0, 300, 1, [100, 200, 300], [3, 5, 7], pooling=True)
    print (conv)
    output = conv(sentence)
    print (output.size())
    sentence2 = Var(torch.rand(1, 1, emb_size))
    output2 = conv(sentence2)
    print(output2.size())


# def test_multi_embedding_chanel():
#     emb_size = 300
#     sentence = Var(torch.rand(24, 1, emb_size))
#     sentence2 = Var(torch.rand(24, 1, emb_size))
#     conv = MultiConvModule(0, 300, 2, [100, 200], [3, 5])
#     print (conv)
#     stacksentences = torch.cat([sentence,sentence2], 1)
#     output = conv(stacksentences)
#     print (output.size())
#
# def test_get_multiconv_statedict():
#     conv = MultiConvModule(0, 300, 2, [100, 200], [3, 5])
#     torch.save(conv.state_dict(), 'conv_state.pth')
#     conv2 = MultiConvModule(0, 300, 2, [100, 200], [3, 5])
#     conv2.load_state_dict(torch.load('conv_state.pth'))
#     print ('dog')

