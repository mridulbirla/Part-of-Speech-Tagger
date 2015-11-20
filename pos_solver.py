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
import copy

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
        self.probability_speech = {}
        self.probability_next_speech = {}
        self.probability_word_speech = {}
        self.probability_first_speech = {}

    def posterior(self, sentence, label):
        return 0

    def update(self, m, k, count):
        m[k] = float(m[k]) / count

    # Do the training!
    #
    def train(self, data):
        print "Inside training"
        for line in data:
            for index in range(0, len(line[1])):
                if index ==0:
                    self.probability_first_speech.setdefault(line[1][0],0)
                    self.probability_first_speech[line[1][0]]+=1
                if index < (len(line[1])) - 1:
                    x = line[1][index + 1].lower() + '_' + line[1][index]
                    self.probability_next_speech.setdefault(x, 0)
                    self.probability_next_speech[x] += 1
                x = line[0][index].lower() + '_' + line[1][index]
                self.probability_word_speech.setdefault(x, 0)
                self.probability_word_speech[x] += 1
                self.probability_speech.setdefault(line[1][index], 0)
                self.probability_speech[line[1][index]] += 1
        print len(self.probability_next_speech.keys())
        [self.update(self.probability_word_speech, k, sum(self.probability_word_speech.values())) for k in
         self.probability_word_speech.keys()]
        [self.update(self.probability_first_speech, k, sum(self.probability_first_speech.values())) for k in
         self.probability_first_speech.keys()]
        print "B"
        [self.update(self.probability_next_speech, k, sum(self.probability_next_speech.values())) for k in
         self.probability_next_speech.keys()]
        print "BB"
        [self.update(self.probability_speech, k, sum(self.probability_speech.values())) for k in
         self.probability_speech.keys()]

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
                new_prob=(prob_wrd_spch*self.probability_speech[speech])

                if new_prob>max:
                        s=speech
                        max=new_prob
            if max !=0:
                speech_map=speech_map+[s]
                prob_map=prob_map+[max]
            else:
                speech_map=speech_map+["noun"]
                prob_map=prob_map+[max]
        p=[[speech_map],[]]
        return p

    def mcmc(self, sentence, sample_count):

        sample_list = list(sentence)
        sampled_values = [[]]
        map(lambda x: sampled_values[0].append('noun'), range(0, len(sample_list)))
        print sampled_values
        initial_values = copy.copy(sampled_values[0])
        speech_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        for i in range(0, 500):

            for j in range(0, len(sample_list)):
                probability_values=[]
                for k in range(0, 12):
                    t = 0
                    wrd_speech=1
                    u=1
                    new_speech = speech_list[k].lower()
                    if i == 0:
                        t = self.probability_first_speech[new_speech]
                    else:
                        if new_speech + '_' + initial_values[j - 1] in self.probability_next_speech:
                            t = self.probability_next_speech[new_speech + '_' + initial_values[j - 1]]
                        else:
                            probability_values.append(0)
                            continue
                    x=sample_list[j]+'_'+new_speech
                    if x in self.probability_word_speech:
                        wrd_speech=self.probability_word_speech[x]
                    else:
                        probability_values.append(0)
                        continue

                    if k<len(sample_list)-1:
                        next_speech=initial_values[k+1]+'_'+new_speech
                        if next_speech in self.probability_next_speech.keys() :
                            u=self.probability_next_speech[next_speech]
                        else:
                            probability_values.append(0)
                            continue
                    probability_values.append(u*t*wrd_speech)
                initial_values[j]=speech_list[probability_values.index(max(probability_values))].lower()
            #print initial_values
            sampled_values.append(initial_values)

            top_sampled_value=sampled_values[len(sampled_values)-5:]
        return [top_sampled_value, []]

    def best(self, sentence):
        return [[["noun"] * len(sentence)], []]

    def max_marginal(self, sentence):
        return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

    def viterbi(self, sentence):
        forward_matrix = {}
        # for x in range(12):
        #     for y in range(len(sentence)):
        #         forward_matrix[x,y] = 0

        backward_matrix = {}
        # for x in range(12):
        #     for y in range(len(sentence)):
        #         backward_matrix[x,y] = 0
        path = []
        #filling rest of the matrix
        for index_eachword in range(0,len(sentence)):
            eachword=sentence[index_eachword]
            #initially for 1st word
            if eachword == sentence[0]:
                for speech in ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']:
                    if speech+'_'+sentence[0] in self.probability_word_speech.keys():
                        forward_matrix[speech, eachword] = (self.probability_first_speech(speech) * self.probability_word_speech(speech,'_',sentence[0]))
                    else:
                        forward_matrix[speech, eachword]=0

            else:
                for speech in ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']:
                    max_value = -1
                    if eachword +'_'+speech in self.probability_word_speech.keys():
                            emission=self.probability_word_speech[eachword +'_'+speech]
                    else:
                        forward_matrix[speech, eachword]=0
                        continue
                    for i in ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']:
                        transition_prob=self.probability_next_speech[speech +'_'+ i]
                        prev_word = sentence[index_eachword-1]
                        previous_veterbi_coeff = forward_matrix[i, prev_word]
                        new_veterbi_coeff = transition_prob * previous_veterbi_coeff

                        if new_veterbi_coeff > max_value:
                            max_value = new_veterbi_coeff
                            backward_matrix[speech, eachword] = i
                    forward_matrix[speech, eachword] = max_value*emission

            last_max_value = -1
            if eachword == sentence[len(sentence)-1]:
                for k in ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']:
                    if last_max_value < forward_matrix[k, eachword]:
                        last_max_tag = k

        path.append(last_max_value)

        temp_var = last_max_value
        for i in range (1, len(sentence)):
            path.append(backward_matrix[temp_var, sentence[len(sentence)-i]])
            temp_var= backward_matrix[temp_var, sentence[len(sentence)-i]]



        return [path, []]

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
