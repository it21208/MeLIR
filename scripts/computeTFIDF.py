# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

# The purpose of this script is to compute the TFIDF scores for the synthetic documents of the PubMed topics
import math 
from sklearn.feature_extraction.text import CountVectorizer

try:
    import nltk
except ImportError:
    nltk.download('all')
    import nltk


def computeTFIDF(seedDoc, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict):
    # Initializing some lists.
    tfDict, idfDict, tfidf = ({} for i in range(3))
    docList = [seedDoc.lower().split()]
    
    # N is the document list, in my case it's 1
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

    # Get Term-Frequencies
    for word, count in wordDict.items():

        tfDict[word] = count/float(bowCount)

    for word, val in wordDict.items():

        if val > 0:

            idfDict[word] += 1

    # list of keys to be removed.
    rem_list = []

    for word, val in idfDict.items():

        if word not in idf_word_dict:

            rem_list.append(word)

    # Using pop() plus list comprehension to Remove multiple keys from dictionary.
    [idfDict.pop(key) for key in rem_list]

    [tfDict.pop(key) for key in rem_list]

    idfDict.update(idf_word_dict)

    list_of_words = []

    # Calculate TF-IDF scores for seed document words.
    for word, val in tfDict.items():

        tfidf[word] = val*idfDict[word]

        list_of_words.append(word)

    return(tfidf, list_of_words)
