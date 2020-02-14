# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis
''' The purpose of this script is to compute the TFIDF scores for the synthetic documents of the PubMed topics '''
import math

import nltk
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download('all')


def computeTFIDF(seedDoc, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict):
  tfDict, idfDict, tfidf = ({} for i in range(3))
  docList = [seedDoc.lower().split()]
  # N is the document list, in my case it's 1
  #N = len(docList)
  N = cur_idx
  # Create word dictionary for seedDoc
  wordDict = {}
  my_string = seedDoc.lower().split()

  for item in my_string:

      if item in wordDict:
          wordDict[item] += 1
      else:
          wordDict[item] = 1

  # Create bag of words for seedDoc
  vectorizer = CountVectorizer()
  vectorizer.fit_transform(nltk.word_tokenize(seedDoc)).todense()
  bow = vectorizer.vocabulary_
  bowCount = len(bow)
  idfDict = dict.fromkeys(wordDict, 0)

  #print(wordDict)
  #print(total_words)
  #for word, count in vocab_idx_dict_2.items():

  for word, count in wordDict.items():
    # Get Term-Frequencies
    tfDict[word] = count/float(bowCount)

  for word, val in wordDict.items():
    if val > 0:
      idfDict[word] += 1

  '''
  # Get Inverted Document-Frequencies
  for word, val in idfDict.items():
    idfDict[word] = math.log10(N/float(val))
  '''

  #print(idf_word_dict)
  #print(idfDict)

  # initializing Remove keys 
  rem_list = []

  for word, val in idfDict.items():
    if word not in idf_word_dict:
      #print('yes')
      rem_list.append(word)

  # Using pop() + list comprehension 
  # Remove multiple keys from dictionary 
  [idfDict.pop(key) for key in rem_list]
  [tfDict.pop(key) for key in rem_list] 

  #print(idfDict)
  #print(tfDict)

  idfDict.update(idf_word_dict)
  #print(N)

  '''
  for word, val in idfDict.items():
    idf_dict[word] = math.log10(N/float(val))
  '''

  list_of_words = []
  for word, val in tfDict.items():
    tfidf[word] = val*idfDict[word]
    list_of_words.append(word)

  #print(tfidf)

  return(tfidf, list_of_words)
