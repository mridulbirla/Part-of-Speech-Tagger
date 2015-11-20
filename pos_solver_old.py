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
    def __init__(self):
        self.probability_speech={}
        self.probability_next_speech={}
        self.probability_word_speech={}

    def posterior(self, sentence, label):
        return 0

    def update(self, m, k, count):
        m[k] = float(m[k]) / count

    # Do the training!
    #
    def train(self, data):
        print "Inside training"
        for line in data:
            for index in range(0,len(line[1])):
                if index < (len(line[1]))-1:
                    x=line[1][index+1].lower()+'_'+line[1][index]
                    self.probability_next_speech.setdefault(x,0)
                    self.probability_next_speech[x]+=1
                x=line[0][index].lower()+'_'+line[1][index]
                self.probability_word_speech.setdefault(x,0)
                self.probability_word_speech[x] +=1
                self.probability_speech.setdefault(line[1][index],0)
                self.probability_speech[line[1][index]] += 1
        print "BC"
        [self.update(self.probability_word_speech, k, sum(self.probability_word_speech.values())) for k in self.probability_word_speech.keys()]
        print "B"
        [self.update(self.probability_next_speech, k, sum(self.probability_next_speech.values())) for k in self.probability_next_speech.keys()]
        print "BB"
        [self.update(self.probability_speech, k, sum(self.probability_speech.values())) for k in self.probability_speech.keys()]
    # Functions for each algorithm.
    #
    def naive(self, sentence):
        speech_map=[]
        prob_map=[]
        flag=0
        for word in sentence:
            max=0
            s=''
            flag=1
            if  word=="calmly":
                print "a"
            for speech in self.probability_speech.keys():
                k=word+'_'+speech
                if k not in self.probability_word_speech.keys():
                    continue
                prob_wrd_spch= self.probability_word_speech[k]
                sum=0.0
                # for speech2 in self.probability_speech.keys():
                #     k=word+'_'+speech2
                #     if k not in self.probability_word_speech.keys():
                #         continue
                #     prob_wrd_spch= self.probability_word_speech[k]
                #     new_prob=0.0
                #     sum+=(prob_wrd_spch*self.probability_speech[speech2])
                #     if sum!=0:
                new_prob=(prob_wrd_spch*self.probability_speech[speech])

                if new_prob>max:
                    s=speech
                    max=new_prob
            if s=='':
                speech_map=speech_map+['noun']
                prob_map=prob_map+[1]
            else:
                speech_map=speech_map+[s]
                prob_map=prob_map+[max]

        p=[[speech_map],[prob_map]]
        return p
       # return [[["noun"] * len(sentence)], []]

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
d=[]


s.train(d)
