# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

import operator
import os

import numpy as np


# write results to output file
def writeResults(out_folder, clf, dict_initialScoreRankingResults, LambdaParam, topic_list, list_doc_score):  # list_doc_score_copies

  new_list_doc_score = []

  def normalisation(score):
    s_min, s_max = min(score), max(score)
    temp_var = s_max - s_min
    score[:] = [x - s_min for x in score]
    # Check if I am dividing by zero and handle 
    try:
      score[:] = [x / temp_var for x in score]
    except:
      score[:] = [x for x in score]
    return(score)

  def interpolate(old_score, new_score):
    '''
    print('old_score: ', old_score, len(old_score))
    print('--------')
    print('new_score: ', new_score, len(new_score))
    '''
    old_score = normalisation(old_score)
    new_score = normalisation(new_score)
    
    old_score = [x * (1 - LambdaParam) for x in old_score]
    new_score = [x * LambdaParam for x in new_score]
    score = old_score + new_score
    #score = old_score
    #score = new_score
    #print('------------')
    #print(score)
    return score

  # Use dict_initialScoreRankingResults and list_doc_score

  ''' Iterate through all topics in the topic_list '''
  for idx, topic in enumerate(topic_list):

    old_score, new_score = ([] for i in range(2))

    ''' Loop through the items in the list_doc_score dictionary '''
    for key, value in list_doc_score[idx].items():  # for key, value in (list_doc_score_copies[idx])[idx].items():

      ''' Loop through dict_initialScoreRankingResults for the current topic. '''
      for doc_score_list in dict_initialScoreRankingResults[topic]:

        ''' check if docid is the same in the doc_score_list '''
        if key == doc_score_list[0]:
          old_score.append(doc_score_list[1])
          new_score.append(value)

    float_old_score = [float(i) for i in old_score]

    ''' This is where the proble lies '''
    score = interpolate(np.array(float_old_score), new_score)

    #score = new_score
    #print('------------')
    #print(score)
    #print('score: ', score, len(score))
    ''' I need to update the list_doc_score[idx] with the new_score values for each doc before continuing to write the results. '''
    counter = 0
    # for key, value in (list_doc_score_copies[idx])[idx].items():
    for key, value in list_doc_score[idx].items():
      # ((list_doc_score_copies[idx])[idx])[key] = score[counter]
      try:
        (list_doc_score[idx])[key] = score[counter]
        counter += 1
      except:
        pass

    # sorted_new_doc_score = sorted((list_doc_score_copies[idx])[idx].items(), key=operator.itemgetter(1))
    sorted_new_doc_score = sorted(
        list_doc_score[idx].items(), key=operator.itemgetter(1))
    sorted_new_doc_score.reverse()
    sorted_list_new_doc_score = []
    for key, value in sorted_new_doc_score:
      sorted_list_new_doc_score.append([key, value])
    new_list_doc_score.append(sorted_list_new_doc_score)

  output = out_folder+'clf/'+clf+'/'
  filename = f'pubmed_BMI_CAL' + \
      '.Bm25.b0.75.k1.2.tar.title_and_query'+'.l_'+str(LambdaParam)+'.txt'
  with open(os.path.join(output, filename), 'w') as f:
      for idx, topic in enumerate(topic_list):
        rank = 1
        for docid, score in new_list_doc_score[idx]:
          tag = 'CAL.AutoTAR.Bm25.b0.75.k1.2.tar.title_and_query' + \
              '.l_'+str(LambdaParam)
          f.write(f'{topic} Q0 {docid} {rank} {score} pubmed_{tag}\n')
          rank += 1
