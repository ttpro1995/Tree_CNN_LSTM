from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import utils
import gc
import sys
from meowlogtool import log_util


# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SSTDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer
import numpy as np

# MAIN BLOCK
def main():
    global args
    args = parse_args(type=1)
    print (args.name)
    print (args.model_name)

    if args.mem_dim == 0:
        if args.model_name == 'dependency':
            args.mem_dim = 168
        elif args.model_name == 'constituency':
            args.mem_dim = 150
        elif args.model_name == 'lstm':
            args.mem_dim = 168
        elif args.model_name == 'bilstm':
            args.mem_dim = 168

    if args.num_classes == 0:
        if args.fine_grain:
            args.num_classes = 5 # 0 1 2 3 4
        else:
            args.num_classes = 3 # 0 1 2 (1 neutral)
    elif args.num_classes == 2:
        # assert False # this will not work
        assert not args.fine_grain

    args.cuda = args.cuda and torch.cuda.is_available()
    # args.cuda = False
    print(args)
    # torch.manual_seed(args.seed)
    # if args.cuda:
        # torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data,'vocab-cased.txt') # use vocab-cased
    # build_vocab(token_files, vocab_file) NO, DO NOT BUILD VOCAB,  USE OLD VOCAB

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file)
    print('==> SST vocabulary size : %d ' % vocab.size())

    # Load SST dataset splits

    is_preprocessing_data = False # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data,'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True

    # dev
    dev_file = os.path.join(args.data,'sst_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data,'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True

    criterion = nn.NLLLoss()
    # initialize model, criterion/loss_function, optimizer

    if args.model_name == 'dependency' or args.model_name == 'constituency':
        model = TreeLSTMSentiment(
                    args.cuda, vocab.size(),
                    args.input_dim, args.mem_dim,
                    args.num_classes, args.model_name, criterion
                )
    elif args.model_name == 'lstm' or args.model_name == 'bilstm':
        model = LSTMSentiment(
                    args.cuda, args.train_subtrees,
                    args.input_dim, args.mem_dim,
                    args.num_classes, args.model_name, criterion
                )

    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()

    if args.cuda:
        model.cuda(), criterion.cuda()


    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_split_token = ' '
    if args.embedding == 'glove':
        emb_torch = 'sst_embed.pth'
        emb_vector = 'glove.840B.300d'
        emb_vector_path = os.path.join(args.glove, emb_vector)
        # assert os.path.isfile(emb_vector_path+'.txt')
    elif args.embedding == 'paragram':
        emb_torch = 'sst_embed_paragram.pth'
        emb_vector = 'paragram_300_sl999'
        emb_vector_path = os.path.join(args.paragram, emb_vector)
        assert os.path.isfile(emb_vector_path+'.txt')
    elif args.embedding == 'paragram_xxl':
        emb_torch = 'sst_embed_paragram_xxl.pth'
        emb_vector = 'paragram-phrase-XXL'
        emb_vector_path = os.path.join(args.paragram, emb_vector)
        assert os.path.isfile(emb_vector_path + '.txt')
    elif args.embedding == 'other':
        emb_torch = 'other.pth'
        emb_vector = args.embedding_other
        emb_vector_path = emb_vector
        emb_split_token = '\t'
        assert os.path.isfile(emb_vector_path + '.txt')
    else:
        assert False

    emb_file = os.path.join(args.data, emb_torch)
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:

        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(emb_vector_path, emb_split_token)
        print('==> Embedding vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(),glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print ('quit program')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    embedding_model.state_dict()['weight'].copy_(emb)

    if args.optim=='adam':
        optimizer   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adam_combine':
        optimizer = optim.Adam([
                {'params': model.parameters(), 'lr':args.lr, 'weight_decay':args.wd },
                {'params': embedding_model.parameters(), 'lr': args.emblr, 'weight_decay':args.embwd}
            ])
        args.manually_emb = 0
    elif args.optim == 'adagrad_combine':
        optimizer = optim.Adagrad([
                {'params': model.parameters(), 'lr':args.lr, 'weight_decay':args.wd },
                {'params': embedding_model.parameters(), 'lr': args.emblr, 'weight_decay':args.embwd}
            ])
        args.manually_emb = 0
    elif args.optim == 'adam_combine_v2':
        model.embedding_model = embedding_model
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        args.manually_emb = 0
    metrics = Metrics(args.num_classes)
    utils.count_param(model)

    # create trainer object for training and testing
    if args.model_name == 'dependency' or args.model_name == 'constituency':
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
    elif args.model_name == 'lstm' or args.model_name == 'bilstm':
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)

    trainer.set_initial_emb(emb)

    # trainer = SentimentTrainer(args, model, embedding_model ,criterion, optimizer)

    test_idx_dir = os.path.join(args.data, args.test_idx)
    test_idx = None
    if os.path.isfile(test_idx_dir):
        print('load test idx %s' % (args.test_idx))
        test_idx = np.load(test_idx_dir)

    mode = args.mode
    if mode == 'DEBUG':
        for epoch in range(args.epochs):
            # print a tree
            tree, sent, label = dev_dataset[3]
            utils.print_span(tree, sent, vocab)
            quit()

            dev_loss = trainer.train(dev_dataset)
            dev_loss, dev_pred, _ = trainer.test(dev_dataset)
            test_loss, test_pred, _ = trainer.test(test_dataset)

            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Dev loss   : %f \t' % dev_loss, end="")
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
    elif mode == "PRINT_TREE":
        file_path = os.path.join('print_tree',args.name+'.npy')
        print_list = np.load(file_path)
        utils.print_trees_file_v2(args, vocab, test_dataset, print_list, name='tree')
        print('break')
        quit()
    elif mode == 'EVALUATE':
        filename = args.name + '.pth'
        epoch = args.epochs
        model_name = str(epoch)+'_model_'+filename
        embedding_name = str(epoch)+'_embedding_'+filename
        model = torch.load(os.path.join(args.saved, model_name))
        embedding_model = torch.load(os.path.join(args.saved, embedding_name))

        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        test_loss, test_pred, subtree_metrics = trainer.test(test_dataset, test_idx)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels, test_idx)
        print('Epoch with max dev:' + str(epoch) + ' |test percentage '+ str(test_acc))
        print ('____________________'+str(args.name)+'___________________')
        print_list = subtree_metrics.print_list
        torch.save(print_list, os.path.join(args.saved, args.name + 'printlist.pth'))
        utils.print_trees_file(args, vocab, test_dataset, print_list, name='tree')
    elif mode == "EXPERIMENT":
        # dev_loss, dev_pred = trainer.test(dev_dataset)
        # dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels, num_classes=args.num_classes)
        max_dev = 0
        max_dev_epoch = 0
        filename = args.name + '.pth'
        for epoch in range(args.epochs):
            train_loss_while_training = trainer.train(train_dataset)
            train_loss, train_pred, _ = trainer.test(train_dataset)
            dev_loss, dev_pred, _ = trainer.test(dev_dataset)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels, num_classes=args.num_classes)
            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels, num_classes=args.num_classes)
            print('==> Train loss   : %f \t' % train_loss_while_training, end="")
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            print ('Epoch %d dev percentage %f ' %(epoch, dev_acc))
            print('Train acc %f '%(train_acc))
            if dev_acc > max_dev:
                print ('update best dev acc %f ' %(dev_acc))
                max_dev = dev_acc
                max_dev_epoch = epoch
                utils.mkdir_p(args.saved)
                torch.save(model, os.path.join(args.saved, str(epoch) + '_model_' + filename))
                torch.save(embedding_model, os.path.join(args.saved, str(epoch) + '_embedding_' + filename))
            gc.collect()
        print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
        print('eva on test set ')
        model = torch.load(os.path.join(args.saved, str(max_dev_epoch) + '_model_' + filename))
        embedding_model = torch.load(os.path.join(args.saved, str(max_dev_epoch) + '_embedding_' + filename))
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        test_loss, test_pred, subtree_metrics = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels, num_classes=args.num_classes)
        print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage ' + str(test_acc))
        print_list = subtree_metrics.print_list
        torch.save(print_list, os.path.join(args.saved, args.name + 'printlist.pth'))
        utils.print_trees_file(args, vocab, test_dataset, print_list, name='tree')
        print('____________________' + str(args.name) + '___________________')
    else:
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            train_loss, train_pred, _ = trainer.test(train_dataset)
            dev_loss, dev_pred, _ = trainer.test(dev_dataset)
            test_loss, test_pred, subtree_metrics = trainer.test(test_dataset)

            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            print('Epoch ', epoch, 'test percentage ', test_acc)
            print_list = subtree_metrics.print_list
            torch.save(print_list, os.path.join(args.saved, args.name + 'printlist.pth'))
            utils.print_trees_file(args, vocab, test_dataset, print_list, name='tree')


if __name__ == "__main__":
    args = parse_args(type=1)
    # log to console and file
    logger1 = log_util.create_logger(os.path.join(args.logs,args.name), print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    print ('_________________________________start___________________________________')
    main()
    log_link = log_util.up_gist(os.path.join(args.logs,args.name+'.log'), args.name, __file__, client_id='ec3ce6baf7dad6b7cf2c', client_secret='82240b38a7e662c28b2ca682325d634c9059efb0')
    print(log_link)