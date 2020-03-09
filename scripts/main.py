# !/usr/bin/python
# author = Alexandros Ioannidis

# Copyright Alexandros Ioannidis
# Thesis for Information Retrieval and machine learning for conducting systematic review.
# University of Strathclyde

# Import all necessary libraries.
import collections
import logging
import math
import random
from scipy.sparse import save_npz
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from computeTFIDF import computeTFIDF
from evaluate_topic import evaluate_topic
from load import load
from readFeedbackQRELSintoDict import readFeedbackQRELSintoDict
from addCorrectColumnNumsForSeed import addCorrectColumnNumsForSeed
from readInitialRanking import readInitialRanking
from Sort import Sort
from Sort import Sort_list_using_2nd_element_of_sublists
from negativeValues import negativeValues
from saveQueryTermIDsAndTFIDFValues import saveQueryTermIDsAndTFIDFValues
from sklearn.model_selection import train_test_split
import sys, csv


# Implements the AutoTAR method as displayed in the Diagram: Diagram of the AutoTAR process.png placed in the root folder of this project.
class main:

    # Constructor
    def __init__(self, projDir, seedDoc_filepath, qrels_filepath, tfidf_dict, vocab_idx_dict, topic_docid_label, clf, tfidf_sp, pathToFeatures, docid_idx_dict, topic, qrels_content_filepath):
        self.projDir = projDir
        self.seedDoc_filepath = seedDoc_filepath
        self.qrels_filepath = qrels_filepath
        self.tfidf_dict = tfidf_dict
        self.vocab_idx_dict = vocab_idx_dict
        self.topic_docid_label = topic_docid_label
        self.clf = clf
        self.tfidf_sp = tfidf_sp
        self.pathToFeatures = pathToFeatures
        self.docid_idx_dict = docid_idx_dict
        self.topic = topic
        self.qrels_content_filepath = qrels_content_filepath

    #  1. Construct a synthetic document from the topic description per topic. The initial training set consists of a synthetic document containing Now the topic title (+ processed topic query previously) labeled as “relevant”.
    def constructSyntheticDoc(self, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict):
        logging.info(f'Topic: '+self.topic)
        seedDoc = open(self.projDir+self.seedDoc_filepath, "r").read()

        # Compute TF-IDF score for each word of the Seed doc text.
        computedTFIDF_seedDoc, feature_names = computeTFIDF(seedDoc, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict)

        columns_for_seed_temp, data_seed, list_cols_seed_and_feature_names = ([] for i in range(3))
        for i in range(len(computedTFIDF_seedDoc)):
            columns_for_seed_temp.append(i)

        help_var = [0]*(len(computedTFIDF_seedDoc))
        data = list(computedTFIDF_seedDoc.values())

        # Create sparse matrix for the unique seed document features
        response = csr_matrix((data, (help_var, columns_for_seed_temp)), shape=(len(columns_for_seed_temp), len(help_var)))
        rows_seed, cols_seed = response.nonzero()

        for i, j in zip(*response.nonzero()):
            data_seed.append(response[i, j])

        for idx, i in enumerate(cols_seed):
            try: list_cols_seed_and_feature_names.append([feature_names[idx], i, data[i]])
            except: pass

        return(rows_seed, cols_seed, data_seed, list_cols_seed_and_feature_names)

    def steps2_and_3(self, topic, rows_seed, qrels_filepath, cols_seed, data_seed, qrels_content_filepath, list_cols_seed_and_feature_names, dict_initialScoreRankingResults):
        # 2. The initial training set consists of the seed document identified in step 1, labeled “relevant”, '0' means "non-relevant", '1' means "relevant".
        training_set = collections.OrderedDict()
        # Topic is used as the name-doc num for the seed_doc since the seed document does not have a doc id like the other documents.
        training_set.update({topic: 1})
        # Read the initial ranking.
        list_of_pmids_for_topic = readInitialRanking(self.projDir+'resources/topics/all.topics2017_2018_2019/'+self.topic)
        # Initialise some empty lists.
        d, ordered_feature_names, list_of_pmids_for_topic = ([] for i in range(3))

        # Using improved initial ranking from Lucene4IR RetrievalAppSubset.
        for key, value in dict_initialScoreRankingResults[topic]:
            list_of_pmids_for_topic.append(key)

        # 3. Set the initial batch size B to 1. Alternative initialisation -> round(len(list_of_pmids_for_topic)/3)
        batch = 1
        
        # Build lookup ordered dictionary for pmids I have just read and put corresponding numbering.
        tmp_lst = [n for n in range(len(list_of_pmids_for_topic))]

        lookup_dict = collections.OrderedDict(zip(tmp_lst, list_of_pmids_for_topic))
        lookup_dict.update({len(list_of_pmids_for_topic): topic})
        # Invert the lookup_dict ordered dictionary.
        inverted_lookup_dict = collections.OrderedDict([[v, k] for k, v in lookup_dict.items()])
        # Set right row index for seed sparse matrix to be constructed inside the loop once the cols/features are also set correctly.
        new_rows_seed = np.array([len(list_of_pmids_for_topic)]*len(rows_seed))

        # Order cols and data lists of seed doc.
        for idx, val in enumerate(list(cols_seed)):
            d.append([val, data_seed[idx]])

        ordered_seed = Sort(d)
        ordered_list_cols_seed_and_feature_names = sorted(list_cols_seed_and_feature_names, key=lambda x: int(x[1]))
        cols_seed = [item[0] for item in ordered_seed]
        data_seed = [item[1] for item in ordered_seed]
        for i in ordered_list_cols_seed_and_feature_names:
            ordered_feature_names.append([i[0], i[2]])

        docs_reviewed = set()
        topic_qrels_dict = readFeedbackQRELSintoDict(topic, qrels_filepath)
        topic_qrels_content_dict = readFeedbackQRELSintoDict(topic, qrels_content_filepath)
        doc_score = collections.OrderedDict()
        flag = True
        learning_iterations = 2
        return(topic_qrels_dict, docs_reviewed, cols_seed, data_seed, batch, new_rows_seed, inverted_lookup_dict, doc_score, flag, learning_iterations, list_of_pmids_for_topic, training_set, tmp_lst, lookup_dict, topic_qrels_content_dict, ordered_feature_names)

    def augmentTrainSetWithRandomDocs_splitTraiTest_trainClf(self, list_of_pmids_for_topic, training_set, inverted_lookup_dict, topic, pathToFeatures, tmp_lst, clf, data_seed, new_rows_seed, doc_score, cols_seed, lookup_dict, vocab_idx_dict, tfidf_dict, ordered_feature_names):
        # if len(list_of_pmids_for_topic) > 0 and len(list_of_pmids_for_topic) <= 100: random_pmids = 50 # if len(list_of_pmids_for_topic) > 100 and len(list_of_pmids_for_topic) <= 1500: random_pmids = 300  # if len(list_of_pmids_for_topic) > 1500: random_pmids = 1000

        random_pmids = 100
        if random_pmids >= (len(list_of_pmids_for_topic) // 2):
            random_pmids = len(list_of_pmids_for_topic) // 10

        list_of_pmids_for_topic_temp = [x for x in list_of_pmids_for_topic if x not in list(training_set)]
        random_docs = random.sample(list_of_pmids_for_topic_temp, random_pmids)

        # 4. Temporarily augment the training set by adding random_pmids (100) random documents from the collection temporarily labeled “not relevant” = 0.
        for doc in random_docs:
            training_set.update({doc: 0})

        # Initialise some empty temporary supportive lists.
        training_set_list_of_pmids, X_train, y_train, new_train_set = ([] for i in range(4))

        # Get a list of the PMIDs in the training set.
        training_set_list_of_pmids = [inverted_lookup_dict[i] for i in training_set]

        for key, value in training_set.items():
            X_train.append(key)
            y_train.append(value)

        X_test = np.setdiff1d(list_of_pmids_for_topic, random_docs)

        # Sparse matrix TF-IDF encoding to represent normalized word frequency scores in a vocabulary, reading both (train and test)X & (labels)y. So Train & Test have to split.
        X, y = load(topic, pathToFeatures)
        cx = X.tocoo()

        #save_npz("/home/pfb16181/NetBeansProjects/scr_matrix_CD008782.npz", X)

        train_row, train_col, train_value, test_row, test_col, test_value, y_train, y_test, test_set = ([] for i in range(9))

        for x in np.nditer(X_test):
            test_set.append(str(x))

        for key, value in training_set.items():
            new_train_set.append(key)

        # Build training and testing data with TF-IDF vectors
        y_train_dict = collections.OrderedDict(zip(tmp_lst, [None] * len(list_of_pmids_for_topic)))
        y_test_dict = collections.OrderedDict(zip(tmp_lst, [None] * len(list_of_pmids_for_topic)))

        counter = 0
        for i, j, v in zip(cx.row, cx.col, cx.data):
            try:
                if lookup_dict[i] in new_train_set:
                    y_train_dict[i] = y[i]

                y_test_dict[i] = y[i]
                # Checks if docid is in training or test set.
                if i in training_set_list_of_pmids:
                    train_row.append(i)
                    train_col.append(j)
                    train_value.append(v)

                test_row.append(i)
                test_col.append(j)
                test_value.append(v)
            except Exception as e: pass
            if counter == 0: max_j = j
            if max_j < j: max_j = j
            counter += 1

        y_train = [x for x in list(y_train_dict.values()) if x is not None]
        y_test = [x for x in list(y_test_dict.values()) if x is not None]

        if max(list(np.unique(test_col))) > max(list(np.unique(train_col))):
            temp_var = test_col
            new_cols_seed, N, data_seed = addCorrectColumnNumsForSeed(vocab_idx_dict, ordered_feature_names)
        else:
            temp_var = train_col
            new_cols_seed, N, data_seed = addCorrectColumnNumsForSeed(vocab_idx_dict, ordered_feature_names)

        for i in new_cols_seed:
            if max_j < i: max_j = i

        #saveQueryTermIDsAndTFIDFValues(new_cols_seed, data_seed)

        # Reduce size of rows and data lists by N elements to match the length of the column. The values of the list of rows are assigned the doc num of the topic +1 because it's the seed.
        tmp = new_rows_seed[0]
        new_rows_seed = [tmp]*len(new_cols_seed)
        #data_seed = data_seed[:-N]

        X_train_row = np.array(train_row + list(new_rows_seed))
        X_train_col = np.array(train_col + list(new_cols_seed))
        X_train_value = np.array(train_value + data_seed)

        # Here perhaps I don't have to add the seed dcoument to the test set of docs - play around with the initialisation.
        X_test_row = np.array(test_row + list(new_rows_seed))
        X_test_col = np.array(test_col + list(new_cols_seed))
        x_test_value = np.array(test_value + data_seed)

        col_size_shape = max_j+1
        if max_j+1 < max(list(np.unique(X_train_col)))+1 or max_j+1 < max(list(np.unique(X_test_col)))+1:
            if max(list(np.unique(X_train_col)))+1 < max(list(np.unique(X_test_col)))+1:
                col_size_shape = max(list(np.unique(X_test_col)))+1
            else:
                col_size_shape = max(list(np.unique(X_train_col)))+1

        # Train csr_matrix Must Have The Same Number of features/Cols the Test csr_matrix will have. - # Construct Train csr matrix.
        new_X_train = csr_matrix((X_train_value, (X_train_row, X_train_col)), shape=(max(list(np.unique(X_train_row)))+1, col_size_shape))

        # Construct Test csr matrix.
        new_X_test = csr_matrix((x_test_value, (X_test_row, X_test_col)), shape=(max(list(np.unique(X_test_row)))+1, col_size_shape))

        # Create the y_train labels for new_X_train.
        new_y_train = [0]*(new_X_train.shape[0])
        for key, value in collections.OrderedDict(zip(np.unique(X_train_row), y_train)).items():
            new_y_train[key] = value

        # Set the label of the Seed document to be 1 'relevant'. - The labels should be numbers (0,1).
        new_y_train[-1] = 1

        # 5. Train an SVM classifier with training set. SVM is similar to LR. Apply LR to the train set. SVMlight was used with default parameters.
        y_test, new_p = evaluate_topic(new_X_train, new_y_train, new_X_test, clf)
        y_test = y_test.tolist()
        # save_npz("/home/pfb16181/NetBeansProjects/new_X_train.npz", new_X_train)  #save_npz("/home/pfb16181/NetBeansProjects/new_X_test.npz", new_X_test)

        temp_list, X_testANDy_test = ([] for i in range(2))
        # Combine y_test and X_test in a data structure. 
        try: # Here there might be an error.
            for index, lst in enumerate(y_test):
                temp_list.append([lookup_dict[index], lst])
        # It's the seed document.
        except: pass

        # Get scores for each doc - Using ordered dictionary structure and updating it in each loop.
        for i in temp_list: doc_score.update({i[0]: i[1]})

        # Sort y_test predictions based on the decision function results. 
        #temp_y_test = [y for (p,y) in sorted(zip(new_p, y_test), key=lambda pair: pair[0])]

        # Order X_testANDy_test list based on the decision function or predict_log_proba values
        if new_p is not None:
          temp_y_test = [y for (p, y) in sorted(zip(new_p, temp_list), key=lambda pair: pair[0])]
          temp_y_test.reverse()
          temp_list = temp_y_test
        else:
          # Order X_testANDy_test list based on the 2nd elements of the inner sublists. 
          temp_list = Sort_list_using_2nd_element_of_sublists(temp_list)

        # Remove from X_testANDy_test list the elements from the training_set.
        for i in temp_list:
            # Here i am checking if it's in the testing set by checking if it's not in the training set.
            if i[0] not in list(training_set.keys()): X_testANDy_test.append(i)
        return(X_testANDy_test, doc_score, y_test, random_docs, training_set)

    # Method that implements Continuous Active Learning
    def cal(self, topic_qrels_dict, docs_reviewed, cols_seed, data_seed, batch, new_rows_seed, inverted_lookup_dict, doc_score, flag, learning_iterations, list_of_pmids_for_topic, training_set, topic, pathToFeatures, clf, tmp_lst, lookup_dict, topic_qrels_content_dict, flag_doc_rel_abs_content, vocab_idx_dict, tfidf_dict, ordered_feature_names):
        learning_cycles = 1
        # Counts the number of relevant documents retrieved for each topic. Used by the knee stopping method.
        relret = 0
        # if len(list_of_pmids_for_topic) < 100: learning_iterations = 20   #if len(list_of_pmids_for_topic) > 100: learning_iterations = 40  #if len(list_of_pmids_for_topic) > 1000: learning_iterations = 60

        # Repeat steps 4 - 10 until all documents have been screened for ranked evaluation or until sufficient/stopping method.
        while (flag == True) and (learning_cycles <= learning_iterations):
            X_testANDy_test, doc_score, y_test, random_docs, training_set = self.augmentTrainSetWithRandomDocs_splitTraiTest_trainClf(list_of_pmids_for_topic, training_set, inverted_lookup_dict, topic, pathToFeatures, tmp_lst, clf, data_seed, new_rows_seed, doc_score, cols_seed, lookup_dict, vocab_idx_dict, tfidf_dict, ordered_feature_names)
            training_set = self.removeRandomDocs(random_docs, training_set)
            highest_scoring_batch_docs = self.selectHighestScoringBatchSizeDocsy_test(y_test, batch, X_testANDy_test)
            dict_highest_scoring_batch_docs_with_labels, flag_doc_rel_abs_content, relret = self.labelBatchSizeDocs_as_RelevantOrNot(highest_scoring_batch_docs, topic_qrels_dict, topic_qrels_content_dict, flag_doc_rel_abs_content, relret)
            training_set, batch = self.addDocsToTrainSet_increaseB(training_set, batch, list_of_pmids_for_topic, dict_highest_scoring_batch_docs_with_labels)
            docs_reviewed.update(training_set)

            # Check what the classifier scorers are outputting. The scores are positive no negatives(only the RLS classifier in some Learn Iter. identifies some negatives).
            print('learning_cycle No: ', learning_cycles, ', ', 'docs_reviewed: ', len(docs_reviewed), ', ', 'relret: ', relret, ', ', 'Negative values found: ', negativeValues(y_test))

            # Checks if all documents have been screened.
            if (len(docs_reviewed) >= len(list_of_pmids_for_topic)): flag = False

            learning_cycles += 1
            # If no relevant documents have been retrieved and documents reviewed is more than 150 stop learning for topic. (knee-method)  #if (len(docs_reviewed) >= 150) and (relret == 0): flag = False  # Checks if all documents have been screened. #if (len(docs_reviewed) >= len(list_of_pmids_for_topic)) or (len(highest_scoring_batch_docs)+102 >= len(list_of_pmids_for_topic)): flag = False # Checks if all documents have been screened. #if len(docs_reviewed) >= len(list_of_pmids_for_topic) or len(highest_scoring_batch_docs)+102 >= len(list_of_pmids_for_topic) or len(training_set)+102 >= len(list_of_pmids_for_topic): flag = False
        # Returns an ordered dictionary with the PMIDs and the corresponding document score.
        return(doc_score)

    # 6. Remove the random documents added in step 4.
    def removeRandomDocs(self, random_docs, training_set):
        for key in random_docs:
              if key in training_set: del training_set[key]
        return(training_set)
    
    # 7. Select the highest-scoring B (batch_size) documents from the not reviewed document.
    def selectHighestScoringBatchSizeDocsy_test(self, y_test, batch, X_testANDy_test):
        counter = 0
        highest_scoring_batch_docs = []
        if batch < len(y_test):
            try:
                while counter < batch:
                    highest_scoring_batch_docs.append(X_testANDy_test[counter])
                    counter += 1
             # Error in highest_scoring_batch_docs - proceed.
            except: pass
        else:
            # In the end of each topic-iteration the batch size will be greater than the length of the list of PMIDs this will happen before the flag has become false.
            highest_scoring_batch_docs = X_testANDy_test
        return(highest_scoring_batch_docs)

    def labelBatchSizeDocs_as_RelevantOrNot(self, highest_scoring_batch_docs, topic_qrels_dict, topic_qrels_content_dict, flag_doc_rel_abs_content, relret):
        # 8. Label each of the B documents as “relevant” or “not relevant” by consulting: Previous “abstract” assessments supplied by CLEF.
        dict_highest_scoring_batch_docs_with_labels = collections.OrderedDict()
        highest_scoring_batch_docs_only_docs = [
            item[0] for item in highest_scoring_batch_docs]
        if flag_doc_rel_abs_content == False:
            for key, value in topic_qrels_dict.items():
                if key in highest_scoring_batch_docs_only_docs:
                    dict_highest_scoring_batch_docs_with_labels.update(
                        {key: topic_qrels_dict[key]})
                    if topic_qrels_dict[key] == 1 and topic_qrels_content_dict[key] == 1:
                        flag_doc_rel_abs_content = True
                    # If document retrieved (in the highest scoring batch docs set) is relevant increase the corresponding counter.
                    if topic_qrels_dict[key] == 1:
                        relret += 1
        # Implement B method.
        else:
            for key, value in topic_qrels_content_dict.items():
                if key in highest_scoring_batch_docs_only_docs: dict_highest_scoring_batch_docs_with_labels.update({key: topic_qrels_content_dict[key]})
                # If document retrieved (in the highest scoring batch docs set) is relevant increase the corresponding counter.
                if topic_qrels_content_dict[key] == 1: relret += 1
        return(dict_highest_scoring_batch_docs_with_labels, flag_doc_rel_abs_content, relret)

    def addDocsToTrainSet_increaseB(self, training_set, batch, list_of_pmids_for_topic, dict_highest_scoring_batch_docs_with_labels):
        # 9. Add the documents to the training set.
        training_set.update(dict_highest_scoring_batch_docs_with_labels)
        # 10. Increase B by [B/10] [] this means ceiling [1.1]=2
        if batch < len(list_of_pmids_for_topic): batch += math.ceil(batch/10)
        else: batch = batch
        return(training_set, batch)


# The method run_main in the main.py outside the class instantiates an object of the class main & implements it's methods. The parameters for the constructor & methods of the instantiated object are passed from the script run.py
def run_main(projDir, seedDoc_filepath, qrels_filepath, tfidf_dict, vocab_idx_dict, topic_docid_label, clf, tfidf_sp, pathToFeatures, docid_idx_dict, topic, qrels_content_filepath, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict, dict_initialScoreRankingResults):
    # Instantiate an object from the class main.
    x = main(projDir, seedDoc_filepath, qrels_filepath, tfidf_dict, vocab_idx_dict, topic_docid_label, clf, tfidf_sp, pathToFeatures, docid_idx_dict, topic, qrels_content_filepath)
    # The object implements the method constructSyntheticDoc of the class.
    rows_seed, cols_seed, data_seed, list_cols_seed_and_feature_names = x.constructSyntheticDoc(idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict)
    # The object implements the method steps2_and_3 of the class. This implements the 2nd & 3rd step of the AutoTAR method which are not iterative and are executed once for each topic.
    topic_qrels_dict, docs_reviewed, cols_seed, data_seed, batch, new_rows_seed, inverted_lookup_dict, doc_score, flag, learning_iterations, list_of_pmids_for_topic, training_set, tmp_lst, lookup_dict, topic_qrels_content_dict, ordered_feature_names = x.steps2_and_3(topic, rows_seed, qrels_filepath, cols_seed, data_seed, qrels_content_filepath, list_cols_seed_and_feature_names, dict_initialScoreRankingResults)
    flag_doc_rel_abs_content = False
    # The object implements the method cal of the class.
    doc_score = x.cal(topic_qrels_dict, docs_reviewed, cols_seed, data_seed, batch, new_rows_seed, inverted_lookup_dict, doc_score, flag, learning_iterations, list_of_pmids_for_topic, training_set, topic, pathToFeatures, clf, tmp_lst, lookup_dict, topic_qrels_content_dict, flag_doc_rel_abs_content, vocab_idx_dict, tfidf_dict, ordered_feature_names)
    # It returns to run.py an ordered dictionary with the doc scores for each topic.
    return(doc_score)
