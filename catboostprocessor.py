#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
'''
#pylint: disable=C0326,W0232,C1001,R0903,C0111,C0330,E1101,W0107,R0201,F0401,C0103

from __future__ import print_function
import sys
import os
# import pandas as pd
#import resource
import pickle
import gzip
import random
import socket
import csv
import numpy as np
import gc
# from sklearn.metrics import r2_score
from collections import OrderedDict
from datetime import date
import datetime
import multiprocessing
# import numpy.lib.recfunctions as nprec
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import pathlib
import time
import copy



# from sklearn.dummy import DummyRegressor
# from catboost import Pool, CatBoostRegressor
import pathlib
# import catboost

DATA_DIR= '../../data/'  

def isotimelabel():
    return datetime.datetime.now().isoformat()[:19].replace(':','-')

class CatboostProvider4ML(object):
    def __init__(self):
        time_start = datetime.datetime.now()

        self.num_feature_names = []
        self.cat_feature_names = []
        
        self.rows_num = 29989753 
        with open('cg1_indexes', 'r') as f:
            self.cg1_list = [int(s) for s in f.readline().split(',')]    

        with open('cg2_indexes', 'r') as f:
            self.cg2_list = [int(s) for s in f.readline().split(',')]    

        with open('cg3_indexes', 'r') as f:
            self.cg3_list = [int(s) for s in f.readline().split(',')]    

        self.cg1_cols_num = len(set(self.cg1_list)) #452
        self.cg2_cols_num = len(set(self.cg2_list)) #30441
        self.cg3_cols_num = len(set(self.cg3_list)) #53593

        self.cg1_nz = 1197444563
        self.cg2_nz = 27153927
        self.cg3_nz = 40854355

        self.cg1_nz_test = 51384037
        self.cg2_nz_test = 1227403
        self.cg3_nz_test = 2377433
        self.rows_num_test = 1317220

        self.cg1_id2idx = {}
        for i, cg_ in enumerate(self.cg1_list):
            self.cg1_id2idx[cg_] = i      

        self.cg2_id2idx = {}
        for i, cg_ in enumerate(self.cg2_list):
            self.cg2_id2idx[cg_] = i      

        self.cg3_id2idx = {}
        for i, cg_ in enumerate(self.cg3_list):
            self.cg3_id2idx[cg_] = i      

        for idx_ in self.cg1_list:
            self.cat_feature_names.append("cg1_%03d" % idx_)

        for idx_ in self.cg2_list:
            self.cat_feature_names.append("cg2_%05d" % idx_)

        for idx_ in self.cg3_list:
            self.cat_feature_names.append("cg3_%05d" % idx_)

        self.num_feature_data = None
        self.cat_feature_data = None
        self.train_labels = None

        self.num_feature_names = ["timestamp", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "l1", "l2", "C11", "C12"]
        
        return        


    def csv2csrall(self, dsname, decimator=None, part=0):
        time_start = datetime.datetime.now()
        print("start csv2csrall", decimator, part, datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        all_nonzeros = (self.cg1_nz + self.cg2_nz + self.cg3_nz + self.rows_num*len(self.num_feature_names)) 
        if 'test' in dsname:
            all_nonzeros = (self.cg1_nz_test + self.cg2_nz_test + self.cg3_nz_test + self.rows_num_test*len(self.num_feature_names)) 

        data_ = np.zeros(all_nonzeros, dtype='f4')
        rows_ = np.zeros(all_nonzeros, dtype='i4')
        cols_ = np.zeros(all_nonzeros, dtype='i4')
        rownum_ = self.rows_num 

        if decimator:
            rownum_ //= decimator

        labels_ = []
        k = 0
        with open(dsname + '.csv', 'r') as f: 
            _delim = ';'
            if 'test' in dsname:
                _delim = ','
            reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=_delim) 
            rownum_ = 0
            for rownum, row in enumerate(reader):
                if not decimator or (rownum % decimator == part):
                    if "label" in row:
                        labels_.append(float(row["label"]))
                    for j, nf in enumerate(self.num_feature_names):
                        if row[nf] == '':
                            data_[k] = np.nan    
                        else:    
                            data_[k]=float(row[nf])    
                        rows_[k]=rownum_
                        cols_[k]=j
                        k+=1

                    for s in row["CG1"].split(','):
                        if s and int(s) in self.cg1_id2idx:
                            idx_ =  self.cg1_id2idx[int(s)] + len(self.num_feature_names)
                            data_[k]=1
                            rows_[k]=rownum_
                            cols_[k]=idx_
                            k+=1
                    for s in row["CG2"].split(','):
                        if s and int(s) in self.cg2_id2idx:
                            idx_ =  self.cg2_id2idx[int(s)] + self.cg1_cols_num + len(self.num_feature_names)
                            data_[k]=1
                            rows_[k]=rownum_
                            cols_[k]=idx_
                            k+=1
                    for s in row["CG3"].split(','):
                        if s and int(s) in self.cg3_id2idx:
                            idx_ =  self.cg3_id2idx[int(s)] + self.cg1_cols_num + self.cg2_cols_num + len(self.num_feature_names)
                            data_[k]=1
                            rows_[k]=rownum_
                            cols_[k]=idx_
                            k+=1
                    rownum_ += 1            
        # np.save('label.npy', label)
        if 'test' not in dsname:
            label = np.array(labels_, dtype='f4')
            pickle.dump(label, open(dsname + '-label-%d-%d.pickle' % (decimator, part), 'wb'), protocol=4)
            del label
        gc.collect()
        csrall = csr_matrix((data_[:k], (rows_[:k], cols_[:k])), dtype='f4', shape=(rownum_, self.cg1_cols_num + self.cg2_cols_num + self.cg3_cols_num + len(self.num_feature_names)))
        del data_
        del rows_ 
        del cols_ 
        gc.collect()
        # np.save('csrtrain.npy', csrall)
        pickle.dump(csrall, open(dsname + '-features-%d-%d.pickle' % (decimator, part), 'wb'), protocol=4)

        print("end csv2csrall", decimator, part, datetime.datetime.now(), datetime.datetime.now()-time_start)
        pass  


    # def csv2numpy(self):
    #     def generator_data_numfeatures():
    #         label = np.zeros(self.rows_num, dtype='f4')
    #         numdata = np.zeros((self.rows_num, len(self.num_feature_names) ), dtype='f4')
    #         with open('train.csv', 'r') as f: 
    #             reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=';') 
    #             for rownum, row in enumerate(reader):
    #                 for j, nf in enumerate(self.num_feature_names):
    #                     numdata[rownum, j] = float(row[nf])    
    #                 label[rownum] = float(row["label"])
    #         gc.collect()
    #         pickle.dump(numdata, open('numdata.pickle', 'wb'), protocol=4)
    #         pickle.dump(label, open('label.pickle', 'wb'), protocol=4)
    #     generator_data_numfeatures()            
    #     pass  

    def train_xgb(self):
        import xgboost as xgb

        time_start = datetime.datetime.now()
        print("xgb training start", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        y_train = pickle.load(open("train-label-2-0.pickle", "rb"))
        y_test1 = pickle.load(open("train-label-4-1.pickle", "rb"))
        y_test2 = pickle.load(open("train-label-4-3.pickle", "rb"))
        gc.collect()

        print("labels loaded", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        X_train = pickle.load(open("train-features-2-0.pickle", "rb"))
        X_test1 = pickle.load(open("train-features-4-1.pickle", "rb"))
        X_test2 = pickle.load(open("train-features-4-3.pickle", "rb"))
        gc.collect()

        print("CSR loaded", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        print("start learning ", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()


        params = {
            'max_depth': 8,
            'eta': 1, 
            'objective':'reg:logistic',
            'n_jobs': 8,
            'eval_metric': 'logloss', 
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test1, y_test1), (X_test2, y_test2)], 
            verbose=True)

        print("end learning ", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        isolabel = isotimelabel()

        pickle.dump(model, open('xgb-model-' + isolabel + '.pickle', 'wb'), protocol=4)
        gc.collect()        

        X_predict = pickle.load(open("test-data-features-0-0.pickle", "rb"))
        y_predict = model.predict(X_predict)
        np.savetxt('xgb-result-' + isolabel + '.csv', y_predict, delimiter="\n")
        pass

    def train_catboost(self):
        # self.ct = bcolz.open(datadir, mode='r')
        # ct = self.ct
        time_start = datetime.datetime.now()
        print("catboost training start", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        y_train = pickle.load(open("train-label-2-0.pickle", "rb"))
        y_test1 = pickle.load(open("train-label-4-1.pickle", "rb"))
        y_test2 = pickle.load(open("train-label-4-3.pickle", "rb"))
        gc.collect()

        print("labels loaded", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        X_train = pickle.load(open("train-features-2-0.pickle", "rb"))
        X_test1 = pickle.load(open("train-features-4-1.pickle", "rb"))
        X_test2 = pickle.load(open("train-features-4-3.pickle", "rb"))
        gc.collect()

        print("CSR loaded", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        print("start learning ", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        self.ml_params = {
            "loss_function": 'Logloss',        
            #RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq
            "eval_metric": "Logloss",
            "random_strength": 90,
            "boosting_type": "Plain",
            "bootstrap_type": "Bernoulli",
            "od_type": 'Iter',
            "od_wait": 800,
            "depth": 5,
            "learning_rate": 0.2,
            #"learning_rate": 0.1,
            #"iterations": 45,
            "iterations": 10,
        }

        time_start = datetime.datetime.now()
        print("before import catboost", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()
        import catboost
        print("after import catboost", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        def Pool(X, y):
            cols = self.num_feature_names+self.cat_feature_names
            cb = catboost.Pool(
                    X,
                    label=y,
                    feature_names=cols,
                    cat_features=[]
            )
            return cb

        # cbtrain = get_pool(0, self.rows_num-1000) 
        # try:
        #     pickle.dump(cbtrain, open('cbtrain-pool.pickle', 'wb'), protocol=4)
        # except Exception as ex:
        #     print(ex)    

        # cbtest = get_pool(self.rows_num-1000, self.rows_num)

        # model =  catboost.CatBoostRegressor(**self.ml_params)
        model =  catboost.CatBoostClassifier(**self.ml_params)
        self.model = model

        # eval_set_ = [cbtest]
  
        model.fit(Pool(X_train, y_train),
                    eval_set=[Pool(X_train, y_train), Pool(X_test1, y_test1), Pool(X_test2, y_test2)],
                    use_best_model=True,
                    verbose=True)

        print("end learning ", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        pickle.dump(model, open('model-catboost.pickle', 'wb'), protocol=4)
        gc.collect()        

    def report_catboost(self):
        model = pickle.load(open("model-catboost.pickle", "rb"))
        X_predict = pickle.load(open("test-data-features-0-0.pickle", "rb"))
        # X_predict = pickle.load(open("train-features-4-1.pickle", "rb"))
        # y = pickle.load(open("train-label-4-1.pickle", "rb"))
        y_predict = model.predict(X_predict)
        np.savetxt("catboost-result.csv", y_predict, delimiter="\n")

        pass

    def report_xgboost(self):
        model = pickle.load(open("xgb-model.pickle", "rb"))
        X_predict = pickle.load(open("test-data-features-0-0.pickle", "rb"))
        # X_predict = pickle.load(open("train-features-4-1.pickle", "rb"))
        # y = pickle.load(open("train-label-4-1.pickle", "rb"))
        y_predict = model.predict(X_predict)
        np.savetxt("xgb-result.csv", y_predict, delimiter="\n")

        pass

                    
if __name__ == '__main__':
    # for shop_id in [36060299, 25180299]: #  shops:
    cp = CatboostProvider4ML()
    #cp.csv2csr()
    #cp.csv2numpy()
    # cp.csv2csrall('train',2,0)
    # cp.csv2csrall('train',4,1)
    # cp.csv2csrall('train',4,3)
    #cp.csv2csrall('test-data',0)
    cp.train_xgb()
    # cp.train_catboost()
    # cp.report_catboost()
    #cp.report_xgboost()
    #cp.train()
    # cp.train()
    pass



