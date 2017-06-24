from __future__ import print_function

import os, math
import torch
from tree import Tree
from vocab import Vocab
import torch
from meowlogtool import log_util

# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path, split_token=' '):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(split_token)
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(split_token)
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(map(float, contents[1:]))
            idx += 1
    with open(path+'.vocab','w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename,'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile,'w') as f:
        for token in vocab:
            f.write(token+'\n')

# mapping from scalar to vector
def map_label_to_target(label,num_classes):
    target = torch.zeros(1,num_classes) # Tensor to zeros
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - label
        target[0][ceil-1] = label - floor
    return target

def map_label_to_target_sentiment(label, num_classes = 3 ,fine_grain = False):
    # num_classes not use yet
    target = torch.LongTensor(1)
    if num_classes == 3:
        target[0] = int(label) # nothing to do here as we preprocess data
    elif num_classes == 2: # so this case have 2 output
        if label == 2:
            target[0] = int(1)
        elif label == 1: # discard all neutral sample
            return None
        else:
            target[0] = int(label)
    return target

def print_tree_file(file_obj, vocab, word, tree, pred_info = None, level = 0):
    """
    Print tree for debug
    :param vocab:
    :param input:
    :param tree:
    :param level:
    :return:
    """
    indent = ''
    leaf_range = len(word)
    for i in range(level):
        indent += '| '
    line = indent + str(tree.idx) + ' '
    idx = tree.idx

    # label
    if tree.gold_label != None:
        line += str(tree.gold_label) + ' '

    # predict info
    if pred_info is not None:
        if tree.idx in pred_info.keys():
            pred = pred_info[tree.idx]
            line += str(pred) + ' '


    # word
    if tree.idx -1 < leaf_range:
        line += str(vocab.idxToLabel[word[tree.idx-1]])+' '

    line += '  ' + '\n'
    file_obj.write(line)
    for i in xrange(tree.num_children):
        print_tree_file(file_obj, vocab, word, tree.children[i], pred_info, level+1)


def print_trees_file(args, vocab, dataset, print_list, name = ''):
    name = name + '.txt'
    treedir = os.path.join(args.logs, args.name)
    folder_dir = treedir
    mkdir_p(treedir)
    treedir = os.path.join(treedir, args.name + name)
    tree_file = open(treedir, 'w')
    incorrect = set()
    for idx in print_list.keys():
        tree_file.write(str(idx) + ' ')
        incorrect.add(idx)
    torch.save(incorrect, os.path.join(folder_dir, args.name + 'incorrect.pth')) # for easy compare
    tree_file.write('\n-----------------------------------\n')
    for idx in print_list.keys():
        tree, sent, label = dataset[idx]
        sent_toks = vocab.convertToLabels(sent, -1)
        sentences = ' '.join(sent_toks)
        tree_file.write('idx_'+str(idx)+' '+sentences+'\n')
        print_tree_file(tree_file, vocab, sent, tree, print_list[idx])
        tree_file.write('------------------------\n')
    tree_file.close()
    tree_dir_link = log_util.up_gist(treedir, args.name, 'tree')
    print('Print tree link '+tree_dir_link)

def print_trees_file_v2(args, vocab, dataset, print_list, name = ''):
    'Print from numpy array'
    name = name + '.txt'
    treedir = os.path.join(args.logs, args.name)
    folder_dir = treedir
    mkdir_p(treedir)
    treedir = os.path.join(treedir, args.name + name)
    tree_file = open(treedir, 'w')
    incorrect = set()
    for idx in print_list:
        tree_file.write(str(idx) + ' ')
        incorrect.add(idx)
    tree_file.write('\n-----------------------------------\n')
    for idx in print_list:
        tree, sent, label = dataset[idx]
        sent_toks = vocab.convertToLabels(sent, -1)
        sentences = ' '.join(sent_toks)
        tree_file.write('idx_'+str(idx)+' '+sentences+'\n')
        print_tree_file(tree_file, vocab, sent, tree)
        tree_file.write('------------------------\n')
    tree_file.close()
    tree_dir_link = log_util.up_gist(treedir, args.name, 'tree')
    print('Print tree link '+tree_dir_link)

def count_param(model):
    print('_param count_')
    params = list(model.parameters())
    sum_param = 0
    for p in params:
        sum_param+= p.numel()
        print (p.size())
    # emb_sum = params[0].numel()
    # sum_param-= emb_sum
    print ('sum', sum_param)
    print('____________')

def print_tree(tree, level):
    indent = ''
    for i in range(level):
        indent += '| '
    line = indent + str(tree.idx)
    print (line)
    for i in xrange(tree.num_children):
        print_tree(tree.children[i], level+1)

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def flatParameters(model):
    """
    flatten param into 1 dimention
    :param model:
    :return:
    """
    params = list(model.parameters())
    one_dim = [p.view(p.numel()) for p in params]
    params = torch.cat(one_dim)
    return params

def print_span(tree, sent, vocab):
    """print all span of tree (for debug)"""
    nodes = tree.depth_first_preorder()
    for node in nodes:
        lo, hi = node.lo, node.hi
        span_vec = sent[lo - 1:hi]
        sent_toks = vocab.convertToLabels(span_vec, -2)
        sentences = ' '.join(sent_toks)
        print (sentences)
