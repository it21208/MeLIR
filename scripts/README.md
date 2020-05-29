<p align="center">
<img src="https://github.com/it21208/MeLIR/blob/master/MeLIR-logo.png" width="360">
</p>

1) Run the following command to install the lightGBM classifier and also install the XGBooster and SGD Classifiers if these are not installed

```
pip install lightgbm 
``` 

> Also install RLScore  

```
git clone https://github.com/aatapa/RLScore.git
python setup.py install
python setup.py install --home=<dir>
python setup.py build_ext --inplace
```

Install any other related dependencies you might be missing.
Execute the command below from inside the root folder of the project. 

```
python scripts/run.py --seedDoc-folder train/seedDocs/ --tfidf-folder train/tfidf/ --qrels-folder qrels/ --l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ --projDir /home/pfb16181/NetBeansProjects/PubMed-CAL-AutoTar/ --topic-list tl --classifier clf
```

>  Where **clf** is one of the following: svm, lgb (lightGBM), sgd, lsvr and more
>  Where **tl** is one of the following: TOPIC_LIST_SMALL, TOPIC_LIST_2017, TOPIC_LIST_2018, TOPIC_LIST_2019, TOPIC_LIST_UWA_UWB, TOPIC_LIST_A_B_RANK_THRESH_NORMAL, TOPIC_LIST_UWA_UWBC
> After the **--projDir** param the user needs to enter his/her **own user path** to the extracted/clone MeLIR project.

An example to run a project

```
python scripts/run.py --seedDoc-folder train/seedDocs/ --tfidf-folder train/tfidf/ --qrels-folder qrels/ --l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ --projDir /home/pfb16181/NetBeansProjects/PubMed-CAL-AutoTar/ --topic-list TOPIC_LIST_2017 --classifier svm
```