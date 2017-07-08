import pytest
from trainer import SentimentTrainer
from config import parse_args
import torch
import torch.optim as optim
import torch.nn as nn
from model import TreeLSTMSentiment

# def test_trainer():
#     embedding_model = nn.Embedding(100, 300)
#     args = parse_args()
#
#     criterion = nn.NLLLoss()
#     model = TreeLSTMSentiment(
#         args.cuda, args.channel,
#         args.input_dim, args.mem_dim,
#         args.num_classes, args.model_name, criterion
#     )
#     optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
#
#     dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
#     torch.save(dev_dataset, dev_file)
#     is_preprocessing_data = True