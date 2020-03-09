# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

# Check if there are any negative values in the predictions.


def negativeValues(y_test):
    count = 0
    for number in y_test:
        if number < 0:
            count += 1
    return(count)
