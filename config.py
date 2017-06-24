import argparse
import random
def parse_args(type=0):
    if type == 0:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
        parser.add_argument('--data', default='data/sick/',
                            help='path to dataset')
        parser.add_argument('--glove', default='data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=15, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.01, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--optim', default='adam',
                            help='optimizer (default: adam)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=True)

        args = parser.parse_args()
        return args
    elif type == 10:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment relatedness')
        parser.add_argument('--name', default='default_name',
                            help='name for log and saved models')
        parser.add_argument('--saved', default='saved_model',
                            help='name for log and saved models')

        parser.add_argument('--model_name', default='dependency',
                            help='model name constituency or dependency')
        parser.add_argument('--data', default='data/sick/',
                            help='path to dataset')
        parser.add_argument('--glove', default='../treelstm.pytorch/data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--paragram', default='/media/vdvinh/25A1FEDE380BDADA/data/john',
                            help='directory with paragram embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=10, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--embedding', default='glove',
                            help='embedding type paragram or glove (default: glove)')
        parser.add_argument('--seed', default=int(random.random()*1e+9), type=int,
                            help='random seed (default: 123)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        cuda_parser.add_argument('--lower', dest='cuda', action='store_true')
        parser.set_defaults(cuda=True)
        parser.set_defaults(lower=True)

        args = parser.parse_args()
        return args
    elif type == 1:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
        parser.add_argument('--name', default='default_name',
                            help='name for log and saved models')
        parser.add_argument('--saved', default='saved_model',
                            help='name for  saved models')
        parser.add_argument('--logs', default='logs',
                            help='name for logs')
        parser.add_argument('--mode', default='EXPERIMENT',
                            help='MODE')


        parser.add_argument('--test_idx', default='test_idx.npy',
                            help='dir to test idx np')

        parser.add_argument('--model_name', default='constituency',
                            help='model name constituency or dependency or lstm or bilstm')
        parser.add_argument('--data', default='data/sst/',
                            help='path to dataset')
        parser.add_argument('--glove', default='../treelstm.pytorch/data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--embedding', default='glove',
                            help='embedding type paragram or paragram_xxl or glove (default: glove)')
        parser.add_argument('--paragram', default='/media/vdvinh/25A1FEDE380BDADA/data/john',
                            help='directory with paragram embeddings')
        parser.add_argument('--embedding_other', default='meow',
                            help='directory with other embeddings')

        parser.add_argument('--mem_dim', default=0, type=int,
                            help='memory dimension (default: 0 auto set)')
        parser.add_argument('--input_dim', default=300, type=int,
                            help='input dimension (default: 300, size of glove embedding)')

        parser.add_argument('--train_subtrees', default=4, type=int,
                            help='number of subtree to sample default 4')


        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=10, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0.1, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--embwd', default=0, type=float,
                            help='weight decay for embedding(default: 0)')
        parser.add_argument('--reg', default=0, type=float,
                            help='l2 regularization (default: 0)')
        parser.add_argument('--embreg', default=0, type=float,
                            help='l2 regularization (default: 0)')

        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--manually_emb', default=1, type=int,
                            help='manually update embedding (default: 1)')

        parser.add_argument('--seed', default=int(random.random()*1e+9), type=int,
                            help='random seed (default: random)')
        parser.add_argument('--fine_grain', default=False, type=bool,
                            help='fine grained (default False)')
        parser.add_argument('--num_classes', default=0, type=int,
                            help='num_classes to classify (default 0, meaning auto depend on fine_grain)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        cuda_parser.add_argument('--lower', dest='cuda', action='store_true')
        parser.set_defaults(cuda=True)
        parser.set_defaults(lower=True)

        args = parser.parse_args()
        return args