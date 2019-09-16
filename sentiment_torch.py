import pandas as pd
import zipfile
import collections
import numpy as np

import string
import math
import random


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time

from inputdata import Options, scorefunction
from model import skipgram


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

class word2vec:
    def __init__(self, inputfile, vocabulary_size=100000, embedding_dim=200, epoch_num=64, batch_size=256, windows_size=5, neg_sample_num=10):
        self.op = Options(inputfile, vocabulary_size)
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num

  # pylint: disable=missing-docstring
  # Function to draw visualization of distance between embeddings.
    def plot_with_labels(self, low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
          x, y = low_dim_embs[i, :]
          plt.scatter(x, y)
          plt.annotate(
              label,
              xy=(x, y),
              xytext=(5, 2),
              textcoords='offset points',
              ha='right',
              va='bottom')

        plt.savefig(filename)
        plt.show()

    def train(self):
        # cudnn.benchmark = True
        model = skipgram(self.vocabulary_size, self.embedding_dim)
        if torch.cuda.is_available():
            print("using cuda")
            model.cuda()
        else:
            print("not using cuda")

        optimizer = optim.SGD(model.parameters(),lr=0.2)
        for epoch in range(self.epoch_num):
            start = time.time()
            self.op.process = True
            batch_num = 0
            batch_new = 0
            while self.op.process:
                pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, self.batch_size)
                loss.backward()

                optimizer.step()

                if batch_num%30000 == 0:
                    torch.save(model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch, batch_num))

                if batch_num%2000 == 0:
                    end = time.time()
                    word_embeddings = model.input_embeddings()
                    # sp1, sp2 = scorefunction(word_embeddings)
                    print('epoch,batch=%2d %5d:  pair/sec = %4.2f loss=%4.3f\r'\
                    %(epoch, batch_num, (batch_num-batch_new)*self.batch_size/(end-start),loss.data),end="")
                    batch_new = batch_num
                    start = time.time()
                batch_num = batch_num + 1
            print()

        tsne = TSNE(perplexity=30, n_components=2, init='random', n_iter=5000)
        embeds = model.u_embeddings.weight.data
        labels = []
        tokens = []
        max_size = 1000
        for idx in range(min(len(embeds), len(self.op.vocab_words), max_size)):
            tokens.append(embeds[idx].cpu().numpy())
            labels.append(self.op.vocab_words[idx])
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(tokens)
        low_dim_embs = tsne.fit_transform(pca_result)
        self.plot_with_labels(low_dim_embs, labels, 'tsne.png')


        print("Optimization finished!")


if __name__ == "__main__":
    lines = pd.read_csv('data/Shakespeare_data.csv')
    lines = lines['PlayerLine']
    words = lines.apply(lambda s: s.translate(str.maketrans("","", string.punctuation))).str.split().apply(pd.Series).stack().reset_index(drop=True)
    wc = word2vec(words)
    wc.train()



