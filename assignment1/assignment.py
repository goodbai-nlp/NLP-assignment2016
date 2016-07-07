#!/usr/bin/env python
# encoding: utf-8
"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: assignment.py
@time: 16-7-6 下午8:27
"""
from __future__ import print_function
from collections import Counter
from math import log
import time

class HMM(object):
    def __init__(self, epsilon=1e-5, training_data=None):
        self.epsilon = epsilon
        self.ALPHA = 0.9
        if training_data is not None:
            self.fit(training_data)

    def fit(self, training_data):
        '''
        Counting the number of unigram, bigram, cooc and wordcount from the training
        data.

        Parameters
        ----------
        training_data: list
            A list of training data, each element is a tuple with words and postags.
        '''
        self.unigram = Counter()  # The count of postag unigram, e.g. unigram['NN']=5
        self.bigram = Counter()  # The count of postag bigram, e.g. bigram[('PRP', 'VV')]=1
        self.cooc = Counter()  # The count of word, postag, e.g. cooc[('I', 'PRP')]=1
        self.wordcount = Counter()  # The count of word, e.g. word['I']=1
        self.tagset = set()
        self.wordset = set()
        print('building HMM model ...')

        for words, tags in training_data:
            # Your code here! You need to implement the ngram counting part. Please count
            self.tagset |= set(tags)
            self.wordset |= set(words)
            # - unigram
            # - wordcount
            self.unigram.update(tags)
            self.wordcount.update(words)
            # - bigram
            for i in range(len(tags) - 1):
                self.bigram.update([(tags[i], tags[i + 1])])
                # print (tags[i],tags[i+1])
            # - cooc
            self.cooc.update([(words[i], tags[i]) for i in range(len(words))])

        print('HMM model is built.')
        self.postags = [k for k in self.unigram]

    def emit(self, words, i, tag):
        '''
        Given a word and a postag, give the log emission probability of P(word|tag)
        Please refer the `foundation of statistial natural language processing`, Chapter 10

        Parameters
        ----------
        words: list(str)
            The list of words
        i: int
            The ith word
        tag: str
            The postag

        Returns
        -------
        prob: float
            The log probability
        '''
        # Your code here! You need to implement the log emission probability part.
        rob = words[i]
        ss = self.unigram[tag]
        # print '混淆概率'+tag+'->'+words[i],self.cooc[(rob,tag)],ss
        # prob = log((self.cooc[rob,tag]+1)/(float(ss)+len(words)))       #拉普拉斯平滑
        # print self.cooc[rob,tag]+1,ss+len(words)
        # prob2 = log(self.wordcount[rob]/float(ss))
        res1 = self.cooc[rob,tag]/float(ss)
        res2 = 1.0/ss
        prob = self.calc(res1,res2)
        return prob

    def trans(self, tag, tag1):
        '''
        Given two postags, give the log transition probability of P(tag1|tag)
        Please refer the `foundation of statistial natural language processing`, Chapter 10

        Parameters
        ----------
        tag: str
            The previous postag
        tag1: str
            The current postag

        Returns
        -------
        prob: float
            The log probability
        '''
        # Your code here! You need to implement the log transition probability part.
        # ss = sum([value for key,value in self.bigram.items() if key[1]== tag])
        ss = float(self.unigram[tag])
        # print '转移概率'+tag+"->"+tag1,self.bigram[(tag,tag1)] , ss
        # prob = log((self.bigram[(tag,tag1)]+1)/float(ss+len(self.postags)))
        # print self.bigram[(tag,tag1)]+1,ss+len(self.postags)
        res1 = self.bigram[(tag,tag1)]/ss
        res2 = 1.0/ss
        prob = self.calc(res1,res2)
        return prob
    def calc(self,res1,res2):
        return log(self.ALPHA*res1+(1-self.ALPHA)*res2)

def viterbi(words, hmm):
    '''
    Viterbi algorihtm.

    Parameters
    ----------
    words: list(str)
        The list of words
    hmm: HMM
        The hmm model

    Return
    ------
    result: list(str)
        The POS-tag for each word.
    '''
    # unpack the length of words, and number of postags
    N, T = len(words), len(hmm.postags)

    # allocate the decode matrix
    score = [[-float('inf') for j in range(T)] for i in range(N)]
    path = [[-1 for j in range(T)] for i in range(N)]

    for i, word in enumerate(words):
        if i == 0:
            for j, tag in enumerate(hmm.postags):
                score[i][j] = hmm.emit(words, i, tag)
        else:
            for j, tag in enumerate(hmm.postags):
                best, best_t = -1e20, -1
                # Your code here, enumerate all the previous tag
                (best,best_t) = max([(score[i-1][y0] + hmm.trans(tag2,tag) + hmm.emit(words,i,tag),y0) for y0,tag2 in enumerate(hmm.postags) if score[i-1][y0]>-1e20])
                score[i][j] = best
                path[i][j] = best_t

    #
    best, best_t = -1e20, -1
    for j, tag in enumerate(hmm.postags):
        if best < score[len(words) - 1][j]:
            best = score[len(words) - 1][j]
            best_t = j

    result = [best_t]
    for i in range(len(words) - 1, 0, -1):
    # Your code here, back trace to recover the full viterbi decode path
        result.append(path[i][result[-1]])
    # convert POStag indexing to POStag str
    result = [hmm.postags[t] for t in reversed(result)]
    return result

training_dataset = [(['dog', 'chase', 'cat'], ['NN', 'VV', 'NN']),
                    (['I', 'chase', 'dog'], ['PRP', 'VV', 'NN']),
                    (['cat', 'chase', 'mouse'], ['NN', 'VV', 'NN'])
                    ]


hmm = HMM(training_data=training_dataset)

# Testing if the parameter are correctly estimated.
# assert hmm.unigram['NN'] == 5
# # print hmm.bigram['VV','NN']
# assert hmm.bigram['VV', 'NN'] == 3
# assert hmm.bigram['NN', 'VV'] == 2
# assert hmm.cooc['dog', 'NN'] == 2
# print hmm.emit(list(hmm.wordset),1,'NN')
# print log(0.7)
# print hmm.trans('VV','NN')
# testing_dataset = [['dog', 'chase', 'mouse'],
#                    ['I', 'chase', 'dog']]
#
# for testing_data in testing_dataset:
#     tags = viterbi(testing_data, hmm)
#     print (tags)
# delta[t,i] = self.B[i,O[t]]*np.array([delta[t-1,j]*self.A[j,i]  for j in range(self.N)]).max()
# (prob, state) = max(
#     [(V[t - 1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states if
#      V[t - 1][y0] >= 0])

from dataset import read_dataset
print (time.strftime('%Y-%m-%d %H:%M:%S'))
train_dataset = read_dataset('./penn.train.pos.gz')
devel_dataset = read_dataset('./penn.devel.pos.gz')

print('%d is training sentences.' % len(train_dataset))
print('%d is development sentences.' % len(devel_dataset))

hmm.fit(train_dataset)
# print("DONE!!")
for x in range(70,100,2):
    hmm.ALPHA = x/100.0
    n_corr, n_total = 0, 0
    case = 0
    for devel_data_x, devel_data_y in devel_dataset:
        # print ('case:',case)
        # print devel_data_x
        pred_y = viterbi(devel_data_x, hmm)
        case+=1
        # print pred_y
        # print devel_data_y
        for pred_tag, corr_tag in zip(pred_y, devel_data_y):
            if pred_tag == corr_tag:
                n_corr += 1
            n_total += 1

    print("accuracy=%f, alpha = %f" % (float(n_corr)/ n_total,hmm.ALPHA))
print (time.strftime('%Y-%m-%d %H:%M:%S'))
# print (viterbi(['HMM', 'is', 'a', 'widely', 'used', 'model', '.'], hmm))
# print (viterbi(['I', 'like', 'cat', ',', 'but', 'I', 'hate', 'eating', 'fish', '.'], hmm))


# test_dataset = read_dataset('./penn.test.pos.blind.gz')
#
# fpo=open('./penn.test.pos.out', 'w')
# case = 0
# for test_data_x, test_data_y in test_dataset:
#     pred_y = viterbi(test_data_x, hmm)
#     print(" ".join(y for y in pred_y), file=fpo)
#
# fpo.close()