# -*- coding: utf-8 -*-
###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
#
#
#
#
#
##
####

import random
import math
from collections import Counter
import itertools
import operator


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    def update(self, m, k, count):
        m[k] = round(m[k] / count, 10)

    # Do the training!
    #
    def train(self, data):
        test_data = [line.rstrip(' \n').split(" ") for line in open("bc.train", "r")]
        word_speech={}
        next_speech={}
        for line in test_data:
            for i in range(0,len(line)-1,2):
                x=line[i].lower()+'_'+line[i+1]
                word_speech.setdefault(x,0)
                word_speech[x]+=1

        for line in test_data:
            for i in range(1,len(line)-2,2):
                x=line[i+2]+'_'+line[i]
                next_speech.setdefault(x,0)
                next_speech[x]+=1
        merged_test_data = list(itertools.chain.from_iterable(test_data))
        # p=list(str(",. (*&^%$£@!/`'\?][=-#¢"))
        # p+=['','\n','``','\'\''] #,'NOUN','VERB','ADV','CONJ','DET','PRT','NUM','ADJ','X','PRON','ADP']
        part_of_speech = ['NOUN', 'VERB', 'ADV', 'CONJ', 'DET', 'PRT', 'NUM', 'ADJ', 'X', 'PRON', 'ADP', '.']
        # new_list=[x for x in merged if x in p]
        new_list = []
        for i in range(0, len(test_data)):
            if not (i != 0 and (merged_test_data[i] == '.' and merged_test_data[i - 1] == merged_test_data[i])) and \
                            merged_test_data[i] in part_of_speech:
                new_list.append(merged_test_data[i])

        count = Counter(map(lambda x: x.lower(), new_list))
        m = dict(count)
        count123 = float(sum(m.values()))
        [self.update(m, k, count123) for k in m.keys()]
        #  count=Counter(new_list)
        count123 = float(sum(word_speech.values()))
        [self.update(word_speech, k, count123) for k in word_speech.keys()]
        count123 = float(sum(next_speech.values()))
        [self.update(next_speech, k, count123) for k in next_speech.keys()]

        print count

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        return [[["noun"] * len(sentence)], []]

    def mcmc(self, sentence, sample_count):
        return [[["noun"] * len(sentence)] * sample_count, []]

    def best(self, sentence):
        return [[["noun"] * len(sentence)], []]

    def max_marginal(self, sentence):
        return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

    def viterbi(self, sentence):
        return [[["noun"] * len(sentence)], []]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algorithm!"


s = Solver()
d = []
s.train(d)
