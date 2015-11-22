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
        self.all_word_speech_keys=[]
        self.all_next_speech_keys=[]
    def posterior(self, sentence, label):
        return 0

    def update(self, m, k, count):
        m[k] = float(m[k]) / count

    # Do the training!
    #
    def train(self, data):
        print "Inside training"
        total_count=0
        for line in data:
            total_count+=len(line[1])
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

        #s=min(self.probability_next_speech.values())
        s=1
        self.probability_next_speech['x_pron'] = 1
        [self.update(self.probability_next_speech, k, self.probability_speech[k.split('_')[1]]) for k in
         self.probability_next_speech.keys()]

        [self.update(self.probability_word_speech, k, self.probability_speech[k.split('_')[1]]) for k in
         self.probability_word_speech.keys()]

        [self.update(self.probability_speech, k, total_count) for k in
         self.probability_speech.keys()]
        count_first_speech=sum(self.probability_first_speech.values())
        [self.update(self.probability_first_speech, k, count_first_speech) for k in
         self.probability_first_speech.keys()]
        self.all_next_speech_keys=self.probability_next_speech.keys()
        self.all_word_speech_keys=self.probability_word_speech.keys()
        print "Training Complete"
    # Functions for each algorithm.
    #
    def naive(self, sentence):
        speech_map=[]
        prob_map=[]
        '''
        for word in sentence:
            max=0
            s=''

            for speech in self.probability_speech.keys():
                k=word+'_'+speech
                if k not in self.probability_word_speech.keys():
                    continue
                prob_wrd_spch= self.probability_word_speech[k]
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
        '''
        return [[["noun"] * len(sentence)], []]

    def mcmc(self, sentence, sample_count):
        
        sample_list = list(sentence)
        sampled_values = [[]]
        map(lambda x: sampled_values[0].append('noun'), range(0, len(sample_list)))
        initial_values = copy.copy(sampled_values[0])
        speech_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        for i in range(0, 200):

            for j in range(0, len(sample_list)):
                probability_values=[]
                for k in range(0, 12):
                    t = 0
                    wrd_speech=1
                    u=1
                    new_speech = speech_list[k]
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
                initial_values[j]=speech_list[probability_values.index(max(probability_values))]
            sampled_values.append(initial_values)
            top_sampled_value=sampled_values[len(sampled_values)-5:]

        return [top_sampled_value, []]

        #return [[["noun"] * len(sentence)], []]

    def best(self, sentence):
        return [[["noun"] * len(sentence)], []]

    def max_marginal(self, sentence):
        return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

    def viterbi(self, sentence):
        forward_matrix = {}
        path = []
        speech_list=['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        previous_flag={}
        for index_eachword in range(0,len(sentence)):
            eachword=sentence[index_eachword]
            last_max_value = 0
            if eachword[len(eachword)-1]=='s' and eachword[len(eachword)-2]=="'":
                eachword=eachword[0:len(eachword)-2]
            previous_flag[index_eachword]=0
            if index_eachword == 0:
                for speech in speech_list:
                    if sentence[0]+'_'+speech in self.all_word_speech_keys:
                        forward_matrix[speech, index_eachword] = ((self.probability_first_speech[speech] * self.probability_word_speech[sentence[0]+'_'+speech]),-1)
                        previous_flag[index_eachword]=1
                    else:
                        forward_matrix[speech, index_eachword] =(float(4.22316915059e-10) * self.probability_first_speech[speech],-1)
                        previous_flag[index_eachword]=1
                    if index_eachword==len(sentence)-1 and last_max_value <forward_matrix[speech, index_eachword][0]:
                        last_max_value=forward_matrix[speech, len(sentence)-1][0]
                        last_max_tag = speech
            else:
                temp_back_pointer=-1
                for speech in speech_list:
                    max_value = 0
                    if eachword +'_'+speech in self.all_word_speech_keys:
                            emission=self.probability_word_speech[eachword +'_'+speech]
                            previous_flag[index_eachword]=1
                    else:
                            emission=float(4.22316915059e-10)
                            previous_flag[index_eachword]=1
                    for i in speech_list:
                        if speech +'_'+ i in self.all_next_speech_keys:
                            transition_prob=self.probability_next_speech[speech +'_'+ i]
                        else:
                            transition_prob=0
                        previous_veterbi_coeff = forward_matrix[i, index_eachword-1][0]
                        new_veterbi_coeff = float(transition_prob) * previous_veterbi_coeff
                        if new_veterbi_coeff > max_value:
                            max_value = float(new_veterbi_coeff)
                            temp_back_pointer=i
#
                    forward_matrix[speech, index_eachword] = (float(max_value)*emission,temp_back_pointer)
                    if index_eachword==len(sentence)-1 and last_max_value <forward_matrix[speech, index_eachword][0]:
                        last_max_value=forward_matrix[speech, len(sentence)-1][0]
                        last_max_tag = speech
            '''
            last_max_value = 0
            if index_eachword==len(sentence)-1:
                for k in speech_list:

                        if last_max_value < forward_matrix[k, len(sentence)-1][0]:
                            last_max_tag = k

                            #previous_tag=backward_matrix[k,len(sentence)-1]
            '''


        #temp_var = previous_tag
        for i in list(reversed(range(0,len(sentence)))):
            # get max tag used
            path.append(last_max_tag)
            last_max_tag=forward_matrix[last_max_tag, i][1]




        path =list(reversed(path))

        return [[path], []]


       #return [[["noun"] * len(sentence)], []]


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

