#!/usr/bin/env python
# encoding: utf-8
"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: assignment2.py
@time: 16-7-7 下午8:44
"""
from copy import copy


class PerceptronClassifier(object):
    # The perceptron classifier
    def __init__(self, max_iter=10, training_data=None, devel_data=None):
        '''
        Parameters
        ----------
        max_iter: int
            The max number of iteration
        training_data: list
            The training data
        devel_data: list
            The development data, to determine the best iteration.
        '''
        self.max_iter = max_iter
        if training_data is not None:
            self.fit(training_data, devel_data)

    def fit(self, training_data, devel_data=None):
        '''
        Estimate the parameters for perceptron model. For multi-class perceptron, parameters can be
        treated as a T \times D matrix W, where T is the number of labels and D is the number of
        features.
        '''
        # feature_alphabet is a mapping from feature string to it's dimension in the feature space,
        # e.g. feature_alphabet['U1=I']=3, which means 'U1=I' is in the third column of W
        #
        # W = [[ . . 1 . . .],
        #      ...
        #      [ . . 1 . . .]]
        #            ^
        #            |
        #         'U1=I'
        self.feature_alphabet = {'None': 0}
        self.label_alphabet = {}

        # Extract features, build the feature_alphabet, label_alphabet and training instance pairs.
        # Each instance consist a tuple (X, Y) where X is the mapped features (list(int)), and Y is
        # the index of the corresponding label.
        instances = []
        for words, tags in training_data:
            L = len(words)
            prev = '<s>'
            for i in range(L):
                # Your code here, extract features and give it into X, convert POStag to index and
                # give it to Y
                X = self.extract_features(words, i, prev)
                Y = i if tags[i] not in self.label_alphabet.keys() else self.label_alphabet[tags[i]]
                instances.append((X, Y))
                if tags[i] not in self.label_alphabet.keys():
                    self.label_alphabet[tags[i]] = i
                prev = tags[i]

        # Build a mapping from index to label string to recover POStags.
        self.labels = [-1 for k in self.label_alphabet]
        for k in self.label_alphabet:
            self.labels[self.label_alphabet[k]] = k

        self.D, self.T = len(self.feature_alphabet), len(self.label_alphabet)
        print('number of features : %d' % self.D)
        print('number of labels: %d' % self.T)

        # Allocate the weight matrix W
        self.W = [[0 for j in range(self.D)] for i in range(self.T)]
        self.best_W = None
        best_acc = None

        for it in range(self.max_iter):
            # The training part,
            n_errors = 0
            print('training iteration #%d' % it)
            for X, Y in instances:
                # Your code here, ake a prediction and give it to Z
                Z = self._predict(X)
                if Z != Y:
                    # Your code here. If the predict is incorrect, perform the perceptron update
                    n_errors += 1
                    for x in X:

                # The perceptron update part.

            print('training error %d' % n_errors)

            if devel_data is not None:
                # Test accuracy on the development set if provided.
                n_corr, n_total = 0, 0
                for words, tags in devel_data:
                    prev = '<s>'
                    for i in range(len(words)):
                        Z = self.predict(words, i, prev)
                        Y = self.label_alphabet[tags[i]]
                        if Z == Y:
                            n_corr += 1
                        n_total += 1
                        prev = self.labels[Z]
                print('accuracy: %f' % (float(n_corr) / n_total))
                if best_acc < float(n_corr) / n_total:
                    # If this round is better than before, save it.
                    best_acc = float(n_corr) / n_total
                    self.best_W = copy(self.W)

        if self.best_W is None:
            self.best_W = copy(self.W)

    def extract_features(self, words, i, prev_tag=None, add=True):
        '''
        Extract features from words and prev POS tag, if `add` is True, also insert the feature
        string to the feature_alphabet.

        Parameters
        ----------
        words: list(str)
            The words list
        i: int
            The position
        prev_tag: str
            Previous POS tag
        add: bool
            If true, insert the feature to feature_alphabet.

        Return
        ------
        mapped_features: list(int)
            The list of hashed features.
        '''
        L = len(words)
        context = ['<s>' if i - 2 < 0 else words[i - 2],
                   '<s>' if i - 1 < 0 else words[i - 1],
                   words[i],
                   '<e>' if i + 1 >= L else words[i + 1],
                   '<e>' if i + 2 >= L else words[i + 1]]
        raw_features = ['U1=%s' % context[0],
                        'U2=%s' % context[1],
                        'U3=%s' % context[2],
                        'U4=%s' % context[3],
                        'U5=%s' % context[4],
                        'U1,2=%s/%s' % (context[1], context[2]),
                        'U2,3=%s/%s' % (context[2], context[3]),  # Your code here, extract the bigram raw feature,
                        'U3,4=%s/%s' % (context[3], context[4]), # Your code here, extract the bigram raw feature,
                        'U4,5=%s/%s' % (context[4], context[5]),  # Your code here, extract the bigram raw feature,
                        ]

        if prev_tag is not None:
            raw_features.append('B=%s' % prev_tag)

        mapped_features = []
        for f in raw_features:
            if add and (f not in self.feature_alphabet):
                # Your code here, insert the feature string to the feature_alphabet.
                index = len(self.feature_alphabet)
                self.feature_alphabet[f] = index
            # Your code here, map the string feature to index.
            for item in self.feature_alphabet:
                mapped_features[self.feature_alphabet[item]] =
        return mapped_features

    def _score(self, features, t):
        '''
        Calcuate score from the given features and label t

        Parameters
        ----------
        features: list(int)
            The hashed features
        t: int
            The index of label

        Return
        ------
        s: int
            The score
        '''
        # Your code here, compute the score.
        s=0.0
        for i in range(len(features)):
            s += self.W[t][i] * features[i]
        return s

    def _predict(self, features):
        '''
        Calcuate score from the given features and label t

        Parameters
        ----------
        features: list(int)
            The hashed features
        t: int
            The index of label

        Return
        ------
        best_y: int
            The highest scored label's index
        '''
        pred_scores = [self._score(features, y) for y in range(self.T)]
        best_score, best_y = None, None
        # Your code here, find the highest scored class from pred_scores
        for i in range(len(pred_scores)):
            if pred_scores[i]>best_score or not best_score:
                best_score = pred_scores[i],best_y = i
        return best_y

    def predict(self, words, i, prev_tag=None):
        '''
        Make prediction on list of words

        Parameters
        ----------
        words: list(str)
            The words list
        i: int
            The position
        prev_tag: str
            Previous POS tag

        Return
        ------
        y: int
            The predicted label's index
        '''
        X = self.extract_features(words, i, prev_tag, False)
        y = self._predict(X)
        return y