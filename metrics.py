from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from collections import OrderedDict

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        #hack cai nay cho no thanh accuracy
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x -= x.mean()
        x /= x.std()
        y -= y.mean()

        y /= y.std()
        return torch.mean(torch.mul(x,y))

    def mse(self, predictions, labels):
        x = Var(deepcopy(predictions), volatile=True)
        y = Var(deepcopy(labels), volatile=True)
        return nn.MSELoss()(x,y).data[0]

    def sentiment_accuracy_score(self, predictions, labels, test_idx= None, fine_gained = True, num_classes = 3):
        _labels = deepcopy(labels)
        if num_classes == 2:
            _labels[_labels==2] = 1
        labels2 = []
        if test_idx is not None:
            for idx in test_idx:
                labels2.append(_labels[idx])
            _labels = labels2
            _labels = torch.Tensor(_labels)
        correct = (predictions==_labels).sum()
        total = _labels.size(0)
        acc = float(correct)/total
        return acc

class SubtreeMetric():
    def __init__(self):
        self.correct = {}
        self.total = {}
        self.correct_depth = {}
        self.total_depth = {}
        self.checked_depth = {}
        self.current_idx = -1 # current idx in dataset
        self.print_list = OrderedDict() # list of data point need to print

    def reset(self):
        self.correct = {}
        self.total = {}

    def count(self, correct, height):
        if height in self.total.keys():
            self.total[height] +=1
        else:
            self.total[height] = 1
            self.correct[height] = 0
        if correct:
            self.correct[height] += 1

    def count_depth(self, correct, depth, tree_idx, pred):
        if depth in self.total_depth.keys():
            self.total_depth[depth] +=1
        else:
            self.total_depth[depth] = 1
            self.correct_depth[depth] = 0

            # self.checked_depth[depth] = True
        if correct:
            self.correct_depth[depth] += 1
        else: # incorrect
            if self.current_idx in self.print_list.keys():
                self.print_list[self.current_idx][tree_idx] = pred
            else:
                self.print_list[self.current_idx] = {}
                self.print_list[self.current_idx][tree_idx] = pred

    def checkDepth(self, depth):
        """
        Check if that depth exist (for easier debug)
        :param depth: the depth you want to check
        :return:
        """
        if depth not in self.checked_depth.keys():
            self.checked_depth[depth] = True

    def getAcc(self):
        acc = {}
        for height in self.total.keys():
            acc[height] = float(self.correct[height]) / self.total[height]
        return acc

    def getAccDepth(self, start, end = -1):
        if end == -1:
            acc = {}
            for depth in self.total_depth.keys():
                acc[depth] = float(self.correct[depth]) / self.total[depth]
            return acc
        else:
            total = 0
            correct = 0
            acc = {}
            for depth in range(start, end+1):
                if depth in self.total_depth.keys():
                    acc[depth] = float(self.correct_depth[depth]) / self.total_depth[depth]
                    total += self.total_depth[depth]
                    correct += self.correct_depth[depth]
                else:
                    # the depth is missing here
                    self.correct_depth[depth] = 0
                    self.total_depth[depth] = 0
                    acc[depth] = 0
            group_acc = float(correct)/total
            return acc, group_acc

    def printCheckDepth(self, start, end):
        for depth in range(start, end):
            if depth in self.checked_depth:
                print ('depth '+str(depth) + ' - '+ str(self.checked_depth[depth]))

    def printAccDepth(self, start, end = -1):
        acc, group_acc = self.getAccDepth(start, end)
        for key in acc.keys():
            print ('Depth ' + str(key) +' '+ str(self.correct_depth[key]) +'/'+ str(self.total_depth[key]) +' acc ' + str(acc[key]))
