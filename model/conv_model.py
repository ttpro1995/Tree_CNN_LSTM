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
        if self.cudaFlag:
            self.conv = self.conv.cuda()

    def forward(self, sentence):
        """
        Forward function
        :param sentence: sentence embedding matrix (seq_length, 1, emb_dim) 
        :return: (seq_length, 1, n_filter)
        """
        sentence = sentence.unsqueeze(2)
        sentence = torch.transpose(sentence, 0, 2)
        output = self.conv(sentence)
        output = output.squeeze(3)
        output = torch.transpose(output, 0, 2)
        output = torch.transpose(output, 1, 2)
        return output

if __name__ == "__main__":
    emb_size = 300
    sentence = Var(torch.rand(35, 1, emb_size))
    conv = ConvModule(0, emb_size, 200, 11)
    output = conv(sentence)
    print (output.size())
