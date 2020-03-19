# !/usr/bin/python
# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

# Passing the -v option to test script will instruct unittest.main() to enable a higher level of verbosity. python - v unittest
# The unittest module can be used from the command line to run tests from modules, classes or even individual test methods
# For example
# python - m unittest test_module1 test_module2
# python - m unittest test_module.TestClass
# python - m unittest test_module.TestClass.test_method


import unittest
import json
from os import listdir
from os.path import isfile, join
import myTestSuite
import time

# from test import autotest


class tests():
    print('   ########### Begin Testing Project ###########')
    # initialise variable that will count the number of tests executed
    count_tests = 0

    def test_type_returned(dicts_list):
        count_errors = 0
        if isinstance(dicts_list[0], dict) == True:
            print('True: tfidf_dict is a dictionary')
        else:
            print('False: ', dicts_list[0], ' is not a dictionary')
            count_errors += 1

        if isinstance(dicts_list[1], dict) == True:
            print('True: docid_idx_dict is a dictionary')
        else:
            print('False: ', dicts_list[1], ' is not a dictionary')
            count_errors += 1

        if isinstance(dicts_list[2], dict) == True:
            print('True: vocab_idx_dict is a dictionary')
        else:
            print('False: ', dicts_list[2], ' is not a dictionary')
            count_errors += 1
        return(count_errors)

    # def test_widget_resize():
    #     assertEqual(widget.size(), (50, 50),
    #                      'incorrect default size')
    #     assertEqual(widget.size(), (50, 50),
    #                      'incorrect default size')
    #     assertEqual(widget.size(), (50, 50),
    #                      'incorrect default size')

    # def test_upper(self):
    #     assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    run_test = myTestSuite.run()
    dicts_list = [run_test.tfidf_dict,
                  run_test.docid_idx_dict, run_test.vocab_idx_dict]
    # Alternative method to get dictionaries saved by reading them from file.
    # dicts_filenames = [f for f in listdir('/home/pfb16181/NetBeansProjects/MeLIR/'+'output/src_outputs__test_inputs/dicts/') if isfile(join('/home/pfb16181/NetBeansProjects/MeLIR/'+'output/src_outputs__test_inputs/dicts/', f))]
    # dicts_list = []
    # for dict_filename in dicts_filenames:
    #     s = open('/home/pfb16181/NetBeansProjects/MeLIR/' + 'output/src_outputs__test_inputs/dicts/'+dict_filename, 'r').read()
    #     dicts_list.append(eval(s))

    # ======= Execute test case for to_sparse() method to check if the input params are all dictionaries. ======= #
    try:
        count_tests += 1
        count_errors = test_type_returned(dicts_list)
        print(3-count_errors, ' out of 3 cases from test case ',
              count_tests, 'are okay.')
    except:
        print('An Error was found in test case ', count_tests)


if __name__ == '__main__':
    start_time = time.time()
    count_tests = tests()

    print('----------------------------------------------------------------------')
    if tests.count_tests == 1:
        print('Ran ', tests.count_tests, ' test case in ',
              time.time() - start_time, ' seconds.')

    else:
        print('Ran ', tests.count_tests, ' test cases in ',
              time.time() - start_time, ' seconds.')
