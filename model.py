import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants
import utils
import numpy as np
import math

class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx.cuda()
            self.ox = self.ox.cuda()

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return c, h

class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})    
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()

    def forward(self, lc, lh , rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c =  i* update + lf*lc + rf*rc
        h = F.tanh(c)
        return c, h






class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, training = False, metric = None):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        if tree.num_children == 0:
            # leaf case
            tree.state = self.leaf_module.forward(embs[tree.idx-1])
        else:
            for idx in xrange(tree.num_children):
                _, child_loss = self.forward(tree.children[idx], embs, training, metric)
                loss = loss + child_loss
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(lc, lh, rc, rh)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[1], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
            if not training and metric is not None:
                # if self.args.num_classes == 3:
                output[:, 1] = -9999  # no need middle (neutral) value
                val, pred = torch.max(output, 1)
                pred_cpu = pred.data.cpu()[0][0]
                correct = pred_cpu == tree.gold_label
                metric.count_depth(correct, tree.depth(), tree.idx, pred_cpu)
        return tree.state, loss


    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh

###################################################################

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # self.emb = nn.Embedding(vocab_size,in_dim,
        #                         padding_idx=Constants.PAD)
        # torch.manual_seed(123)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)

        self.criterion = criterion
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params


    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h,1),0)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs),1)
        f = F.torch.cat([self.fh(child_hi)+fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f,1)
        fc = F.torch.squeeze(F.torch.mul(f,child_c),1)

        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o, F.tanh(c))

        return c,h

    def forward(self, tree, embs, training = False, metric = None):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], embs, training, metric)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1], child_c, child_h)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[1], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
            if not training and metric is not None:
                # if self.args.num_classes == 3:
                output[:, 1] = -9999  # no need middle (neutral) value
                val, pred = torch.max(output, 1)
                pred_cpu = pred.data.cpu()[0][0]
                correct = pred_cpu == tree.gold_label
                metric.count_depth(correct, 0, tree.idx, pred_cpu)
        return tree.state, loss

    def get_child_states(self, tree):
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in xrange(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h

##############################################################################
# similarity
class SimilarityModule(nn.Module):
    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(SimilarityModule, self).__init__()
        super(SimilarityModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        if self.cudaFlag:
            self.wh = self.wh.cuda()
            self.wp = self.wp.cuda()
            self.logsoftmax = self.logsoftmax.cuda()

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = self.logsoftmax(self.wp(out))
        return out

class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.childsumtreelstm = ChildSumTreeLSTM(cuda, in_dim, mem_dim, criterion=None)
        self.similarity = SimilarityModule(cuda, mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        lstate, lloss = self.childsumtreelstm(ltree, linputs)
        rstate, rloss = self.childsumtreelstm(rtree, rinputs)
        lh = lstate[1]
        rh = rstate[1]
        output = self.similarity(lh, rh)
        return output



##############################################################################

# output module
class SentimentModule(nn.Module):
    def __init__(self, cuda, mem_dim, num_classes, dropout = False):
        super(SentimentModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = dropout
        # torch.manual_seed(456)
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        if self.cudaFlag:
            self.l1 = self.l1.cuda()

    def forward(self, vec, training = False):
        if self.dropout:
            out = self.logsoftmax(self.l1(F.dropout(vec, training = training)))
        else:
            out = self.logsoftmax(self.l1(vec))
        return out

class TreeLSTMSentiment(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, num_classes, model_name, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.model_name = model_name
        if self.model_name == 'dependency':
            self.tree_module = ChildSumTreeLSTM(cuda, in_dim, mem_dim, criterion)
        elif self.model_name == 'constituency':
            self.tree_module = BinaryTreeLSTM(cuda, in_dim, mem_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def forward(self, tree, inputs, training = False, metric = None):
        tree_state, loss = self.tree_module(tree, inputs, training, metric)
        output = tree.output
        return output, loss

######################################################
class LSTMSentiment(nn.Module):
    def __init__(self, cuda, train_subtrees, in_dim, mem_dim, num_classes, model_name, criterion):
        super(LSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.bidirectional = False
        self.criterion = criterion
        self.train_subtrees = train_subtrees
        self.num_classes = num_classes
        if model_name == 'bilstm':
            self.bidirectional = True
            self.output_module = SentimentModule(cuda, 2*mem_dim, num_classes, dropout=True)
        else:
            self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=mem_dim, bidirectional=self.bidirectional)
        if self.cudaFlag:
            self.lstm = self.lstm.cuda()

    def getParameters(self):
        '''
        flatten parameter
        :return:
        '''
        params = list(self.parameters())
        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, vec, training = False, metric = None):
        '''
        :param tree: tree structure, for subtree sampling
        :param vec: embedding vector of tree
        :param training: training/eval mode
        :param metric: prevent error, no use here
        :return:
        '''
        nodes = tree.depth_first_preorder()
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        if self.train_subtrees == -1:
            n_subtree = len(nodes)
        else:
            n_subtree = self.train_subtrees + 1
        discard_subtree = 0 # trees are discard because neutral
        if training:
            for i in range(n_subtree):
                if i == 0:
                    node = nodes[0]
                elif self.train_subtrees != -1:
                    node = nodes[int(math.ceil(np.random.uniform(0, len(nodes)-1)))]
                else:
                    node = nodes[i]
                lo, hi = node.lo, node.hi
                span_vec = vec[lo-1:hi] # [inclusive, excludsive)
                _, hn = self.lstm.forward(span_vec)
                h = hn[0]
                if self.bidirectional:
                    h = torch.cat(h, 1)
                else:
                    h = torch.squeeze(h, 1)
                output = self.output_module.forward(h, training)

                if training and node.gold_label != None:
                    target = utils.map_label_to_target_sentiment(node.gold_label, self.num_classes)
                    if target is None:
                        discard_subtree += 1
                        continue
                    target = Var(target)
                    if self.cudaFlag:
                        target = target.cuda()
                    loss = loss + self.criterion(output, target)

            loss = loss
            n_subtree = n_subtree -discard_subtree
            return output, loss, n_subtree
        else:
            _, hn = self.lstm.forward(vec)
            h = hn[0]
            if self.bidirectional:
                h = torch.cat(h, 1)
            else:
                h = torch.squeeze(h, 1)
            output = self.output_module.forward(h, training)
            return output, _

