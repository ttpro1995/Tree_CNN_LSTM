import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class ConvModule(nn.Module):
    def __init__(self, cuda, emb_dim, n_filter, kernel_size):
        super(ConvModule, self).__init__()
        self.cudaFlag = cuda
        self.padding_size = kernel_size/2
        self.conv = nn.Conv2d(1, n_filter, (kernel_size, emb_dim), padding=(self.padding_size,0))
        self.in_dropout = nn.Dropout(p=0.5)
        self.out_dropout = nn.Dropout(p=0.1)
        if self.cudaFlag:
            self.conv = self.conv.cuda()
            self.in_dropout = self.in_dropout.cuda()
            self.out_dropout = self.out_dropout.cuda()

    def forward(self, sentence):
        """
        Forward function
        :param sentence: sentence embedding matrix (seq_length, 1, emb_dim) 
        :return: (seq_length, 1, n_filter)
        """
        sentence = self.in_dropout(sentence)
        sentence = sentence.unsqueeze(2)
        sentence = torch.transpose(sentence, 0, 2)
        output = self.conv(sentence)
        output = output.squeeze(3)
        output = torch.transpose(output, 0, 2)
        output = torch.transpose(output, 1, 2)
        output = self.out_dropout(output)
        return output

if __name__ == "__main__":
    emb_size = 300
    sentence = Var(torch.rand(35, 1, emb_size))
    conv = ConvModule(0, emb_size, 200, 11)
    output = conv(sentence)
    print (output.size())
