# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis


def addCorrectColumnNumsForSeed(vocab_idx_dict, ordered_feature_names):
    new_cols_seed, data_seed = ([] for i in range(2))
    N = 0
    #counter = 1
    for idx, i in enumerate(ordered_feature_names):
        temp = vocab_idx_dict.get(i[0])
        # Here I am searching whether the word exists in the vocabulary and if yes I proceed with storing it and the corresponding tfidf value
        if temp is not None:
            new_cols_seed.append(temp)
            data_seed.append(i[1])
        else:
            N += 1
        #  new_cols_seed.append(counter + (max(list(np.unique(test_col)))))
        #  counter += 1
    return(new_cols_seed, N, data_seed)


# Here I might have to use vocab_idx_dict and tfidf_dict in order to find the correct num features and assign these in the lines below
# I need to find the number of the corresponding features and store in new_cols_seed and then comment out carefully
# def addCorrectColumnNumsForSeed(vocab_idx_dict, ordered_feature_names):
#    new_cols_seed = []
#    for i in ordered_feature_names:
#        temp = vocab_idx_dict.get(i)
#        if temp is not None:
#          new_cols_seed.append(temp)
#        else:
#          new_cols_seed.append(counter + 1 + (max(list(np.unique(test_col))))
#    return(new_cols_seed)
