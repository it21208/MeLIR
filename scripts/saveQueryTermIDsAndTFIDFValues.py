# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis


def saveQueryTermIDsAndTFIDFValues(new_cols_seed, data_seed):
    with open('/home/pfb16181/NetBeansProjects/QueryTermIDsAndTFIDFValues.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(new_cols_seed, data_seed))
    f.close()
