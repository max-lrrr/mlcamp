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
import bcolz
import numpy.lib.recfunctions as nprec
from sklearn.metrics import r2_score
from scipy.sparse import csr_matrix
import pathlib
import time
import copy

# from sklearn.dummy import DummyRegressor
# from catboost import Pool, CatBoostRegressor
import pathlib
# import catboost

DATA_DIR= '../../data/'

class CatboostProvider4ML(object):
    def __init__(self, datadir):
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
        
        print("loading start", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()
        return        


    def csv2csr(self):
        def generator_data_catfeatures():
            data_ = np.ones(self.cg1_nz + self.cg2_nz + self.cg3_nz, dtype=np.bool)
            rows_ = np.zeros(self.cg1_nz + self.cg2_nz + self.cg3_nz, dtype='i4')
            cols_ = np.zeros(self.cg1_nz + self.cg2_nz + self.cg3_nz, dtype='i4')
            k = 0
            with open('train.csv', 'r') as f: 
                reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=';') 
                for rownum, row in enumerate(reader):
                    for s in row["CG1"].split(','):
                        if s:
                            idx_ =  self.cg1_id2idx[int(s)]
                            rows_[k]=rownum
                            cols_[k]=idx_
                            k+=1
                    for s in row["CG2"].split(','):
                        if s:
                            idx_ =  self.cg2_id2idx[int(s)] + self.cg1_cols_num
                            rows_[k]=rownum
                            cols_[k]=idx_
                            k+=1
                    for s in row["CG3"].split(','):
                        if s:
                            idx_ =  self.cg3_id2idx[int(s)] + self.cg1_cols_num + self.cg2_cols_num
                            rows_[k]=rownum
                            cols_[k]=idx_
                            k+=1

            cg_csr = csr_matrix((data_, (rows_, cols_)), dtype=np.bool, shape=(self.rows_num, self.cg1_cols_num + self.cg2_cols_num + self.cg3_cols_num))
            gc.collect()
            pickle.dump(cg_csr, open('cg_csr.pickle', 'wb'), protocol=4)
        generator_data_catfeatures()            
        pass  

    def csv2csrall(self):
        all_nonzeros = self.cg1_nz + self.cg2_nz + self.cg3_nz + self.rows_num*len(self.num_feature_names) 
        data_ = np.zeros(all_nonzeros, dtype='f4')
        rows_ = np.zeros(all_nonzeros, dtype='i4')
        cols_ = np.zeros(all_nonzeros, dtype='i4')
        label = np.zeros(self.rows_num, dtype='f4')
        k = 0
        with open('train.csv', 'r') as f: 
            reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=';') 
            for rownum, row in enumerate(reader):
                label[rownum] = float(row["label"])
                for j, nf in enumerate(self.num_feature_names):
                    data_[k]=float(row[nf])    
                    rows_[k]=rownum
                    cols_[k]=j
                    k+=1

                for s in row["CG1"].split(','):
                    if s:
                        idx_ =  self.cg1_id2idx[int(s)] + len(self.num_feature_names)
                        data_[k]=1
                        rows_[k]=rownum
                        cols_[k]=idx_
                        k+=1
                for s in row["CG2"].split(','):
                    if s:
                        idx_ =  self.cg2_id2idx[int(s)] + self.cg1_cols_num + len(self.num_feature_names)
                        data_[k]=1
                        rows_[k]=rownum
                        cols_[k]=idx_
                        k+=1
                for s in row["CG3"].split(','):
                    if s:
                        idx_ =  self.cg3_id2idx[int(s)] + self.cg1_cols_num + self.cg2_cols_num + len(self.num_feature_names)
                        data_[k]=1
                        rows_[k]=rownum
                        cols_[k]=idx_
                        k+=1
        np.save('label.npy', label)
        del label
        gc.collect()
        csrall = csr_matrix((data_, (rows_, cols_)), dtype='f4', shape=(self.rows_num, self.cg1_cols_num + self.cg2_cols_num + self.cg3_cols_num + len(self.num_feature_names)))
        del data_
        del rows_ 
        del cols_ 
        gc.collect()
        np.save('csrtrain.npy', csrall, allow_pickle=False)
        # pickle.dump(cg_csr, open('csrtrain.pickle', 'wb'), protocol=4)
        pass  


    def csv2numpy(self):
        def generator_data_numfeatures():
            label = np.zeros(self.rows_num, dtype='f4')
            numdata = np.zeros((self.rows_num, len(self.num_feature_names) ), dtype='f4')
            with open('train.csv', 'r') as f: 
                reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=';') 
                for rownum, row in enumerate(reader):
                    for j, nf in enumerate(self.num_feature_names):
                        numdata[rownum, j] = float(row[nf])    
                    label[rownum] = float(row["label"])
            gc.collect()
            pickle.dump(numdata, open('numdata.pickle', 'wb'), protocol=4)
            pickle.dump(label, open('label.pickle', 'wb'), protocol=4)
        generator_data_numfeatures()            
        pass  


    def train(self):
        # self.ct = bcolz.open(datadir, mode='r')
        # ct = self.ct

        time_start = datetime.datetime.now()

        with open("labels.pickle", "rb") as fl:
            self.train_labels = pickle.load(fl)
        gc.collect()

        with open("numdata.pickle", "rb") as fl:
            self.num_feature_data = pickle.load(fl)
        gc.collect()

        with open("cg_csr.pickle", "rb") as fl:
            self.cat_feature_data = pickle.load(fl)
        gc.collect()

        self.ml_params = {
            "loss_function": 'RMSE',        
            #RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq
            "eval_metric": "logloss",
            "random_strength": 90,
            "boosting_type": "Plain",
            "bootstrap_type": "Bernoulli",
            "od_type": 'Iter',
            "od_wait": 800,
            "depth": 6,
            "learning_rate": 0.1,
            #"learning_rate": 0.1,
            #"iterations": 45,
            "iterations": 800,
        }

        time_start = datetime.datetime.now()
        print("before import catboost", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()
        import catboost
        print("after import catboost", datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        def get_pool(row_start, row_end):
            cb = catboost.Pool(catboost.FeaturesData(num_feature_data=self.num_feature_data, #[row_start:row_end],
                                        cat_feature_data=self.cat_feature_data, #[row_start:row_end],
                                        num_feature_names=self.num_feature_names,
                                        cat_feature_names=self.cat_feature_names),
                                        self.train_labels #[row_start:row_end]
                                        ) 
            return cb

        cbtrain = get_pool(0, self.rows_num-1000) 
        cbtest = get_pool(self.rows_num-1000, self.rows_num)

        model =  catboost.CatBoostRegressor(**self.ml_params)
        self.model = model

        eval_set_ = [cbtest]

        model.fit(cbtrain,
                    eval_set=eval_set_,
                    use_best_model=True,
                    #use_best_model=False,
                    verbose=True)

        print("end learning for ", week2predict, datetime.datetime.now(), datetime.datetime.now()-time_start)
        time_start = datetime.datetime.now()

        pickle.dump(model, open('model.pickle', 'wb'), protocol=4)
        gc.collect()        
        pass
    
                    
if __name__ == '__main__':
    # for shop_id in [36060299, 25180299]: #  shops:
    cp = CatboostProvider4ML('train-v4.bcolz')
    #cp.csv2csr()
    #cp.csv2numpy()
    cp.csv2csrall()
    # cp.train()
    pass



