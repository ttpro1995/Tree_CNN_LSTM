from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target, map_label_to_target_sentiment
import torch.nn.functional as F
from metrics import SubtreeMetric

class MultiChannelSentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, embedding_model ,criterion, optimizer):
        super(MultiChannelSentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.embedding_models = embedding_model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.emb_params_init = None

    def set_initial_emb(self, emb):
        self.emb_params_init = emb

    # helper function for training
    def train(self, dataset):
        self.model.train()
        for emb_model in self.embedding_models:
            emb_model.train()
            emb_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        # torch.manual_seed(789)
        indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            tree, sent, label = dataset[indices[idx]]
            input = Var(sent)
            target = Var(map_label_to_target_sentiment(label, self.args.num_classes, fine_grain=self.args.fine_grain))
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()

            emb_list = []
            for emb_model in self.embedding_models:
                emb = F.torch.unsqueeze(emb_model(input), 1)
                emb_list.append(emb)
            emb = torch.cat(emb_list, 1) # (seq, channel, embedding_dim)

            if self.args.model_name == 'lstm' or self.args.model_name == 'bilstm':
                output, err, n_subtrees = self.model.forward(tree, emb, training=True)
                if self.args.train_subtrees == -1:
                    n_subtrees = len(tree.depth_first_preorder())
                else:
                    n_subtrees = self.args.train_subtrees
                batch_size = self.args.batchsize * n_subtrees
            else:
                output, err = self.model.forward(tree, emb, training=True)
                batch_size = self.args.batchsize

            err = err / batch_size

            # err = err / self.args.batchsize
            #params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()


            loss += err.data[0] #
            err.backward()
            k += 1
            if k==self.args.batchsize:
                if self.args.manually_emb == 1:
                    if self.args.embwd == 0: # save time on calculate 0 function
                        for emb_model in self.embedding_models:
                            for f in emb_model.parameters():
                                f.data.sub_(f.grad.data * self.args.emblr)
                    else:
                        for emb_model in self.embedding_models:
                            for f in emb_model.parameters():
                                f.data.sub_(f.grad.data * self.args.emblr + self.args.emblr*self.args.embwd*f.data)

                    # https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
                self.optimizer.step()
                for emb_model in self.embedding_models:
                    emb_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset, test_idx = None):
        subtree_metric = SubtreeMetric()
        self.model.eval()
        for emb_model in self.embedding_models:
            emb_model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        predictions = predictions
        indices = xrange(len(dataset))
        if test_idx is not None:
            indices = test_idx
        predictions = torch.zeros(len(indices))
        predictions = predictions
        for i in tqdm(xrange(len(indices)),desc='Testing epoch  '+str(self.epoch)+''):
            idx = indices[i]
            subtree_metric.current_idx = idx
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,self.args.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            # emb = F.torch.unsqueeze(self.embedding_model(input),1)
            emb_list = []
            for emb_model in self.embedding_models:
                emb = F.torch.unsqueeze(emb_model(input), 1)
                emb_list.append(emb)
            emb = torch.cat(emb_list, 1)  # (seq, channel, embedding_dim)
            output, _ = self.model(tree, emb, metric = subtree_metric) # size(1,5)
            err = self.criterion(output, target)
            loss += err.data[0]
            if self.args.num_classes == 3:
                output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            pred_cpu = pred.data.cpu()[0][0]
            predictions[i] = pred_cpu
            correct = pred_cpu == tree.gold_label
            if self.args.model_name == 'lstm' or self.args.model_name == 'bilstm':
                subtree_metric.count_depth(correct, 0, tree.idx, pred_cpu)
               # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions, subtree_metric
