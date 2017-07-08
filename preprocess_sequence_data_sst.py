import pytreebank
import os
from tqdm import tqdm
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def load_from_file(path):
    """
    :param path: path to trees folder
    :return:
    """
    trees = pytreebank.load_sst(path)
    raw_train = trees["train"]
    raw_dev = trees["dev"]
    raw_test = trees["test"]
    return raw_train, raw_dev, raw_test

def parse_dataset(ds, get_span = False, fine_grain = False):
    label = []
    sentences = []

    if (get_span):
        for tree in tqdm(ds):
           for l, sent in tree.to_labeled_lines():
               if not fine_grain: # 0 neg, 1 neutral, 2 pos
                   if l == 2:
                       continue # we also ignore span with neutral too
                   elif l>=3:
                       l = 2 # so 2 is pos
                   elif l<=1:
                       l = 0
               label.append(l)
               sentences.append(sent)
    else:
        for tree in tqdm(ds):
            l, sent = tree.to_labeled_lines()[0]
            if not fine_grain:  # 0 neg, 1 neutral, 2 pos
                if l == 2:
                    continue  # we dont get any neutral, but we assume it exist
                elif l >= 3:
                    l = 2  # so 2 is pos
                elif l <= 1:
                    l = 0
            label.append(l)
            sentences.append(sent)

    return (sentences, label)

def padding_sentence(sentence_token, length=32):
    """
    Padding <EMPTY> before sentence to make fix length sentence
    :param sentence_token: "this is a string".split()
    :param length: default 32
    :return: list of word
    """
    words = sentence_token[:length]
    if (len(words) < length):
        words = ['XEMPTYX'] * (length - len(words)) + words
    return words

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    data_path = './data/sst_seq'
    train, dev, test = load_from_file('./data/sst_seq')
    for ds, name in zip([train, dev, test], ['train', 'dev', 'test']):
        ds_dir = os.path.join(data_path, name)
        make_dirs(ds_dir)
        file_writer = open(os.path.join(ds_dir, 'sents.toks'), 'w')
        label_writer = open(os.path.join(ds_dir, 'labels.txt'), 'w')
        get_span = False
        if name == 'train':
            get_span = True
        sent_span, label = parse_dataset(ds, get_span, fine_grain=False)
        for sent in  tqdm(sent_span):
            # s = padding_sentence(sent.split())
            s = sent.split()
            file_writer.write(' '.join(s))
            file_writer.write('\n')
        for l in tqdm(label):
            label_writer.write(str(l))
            label_writer.write('\n')

        file_writer.close()
        label_writer.close()



if __name__ == "__main__":
    main()