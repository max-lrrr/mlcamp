#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
'''
#pylint: disable=C0326,W0232,C1001,R0903,C0111,C0330,E1101,W0107,R0201,F0401,C0103
from __future__ import print_function
import sys
import os
#import pandas as pd
try:
    import resource
except:
    pass
import psutil
import gc
import pickle

import numpy as np

def print_memory_info(msg):
    try:
        print("====" + msg + "====")
        process = psutil.Process(os.getpid())
        print(int(process.memory_info().rss/1024/1024), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    except:
        pass
    pass    


#     Z_valid = cs.sample(1000)
# #    Z_valid = cs[cs.week_num.between(MAX_WEEK-52*2, MAX_WEEK-52)]
#     X_valid = Z_valid.drop(['amount'], axis=1)
#     Y_valid = Z_valid['amount']

    print_memory_info("before gc.collect")
    del cs
    del Z_train 
    gc.collect()

    print_memory_info("before model.fit")

    # model.fit(X_train, Y_train,
    #          eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    #          cat_features=[1,2,3,4,5,6],
    #          #use_best_model=True,
    #          #early_stopping_rounds
    #          verbose=True)

    model.fit(X_train, Y_train,
             eval_set=[(X_train, Y_train)],
             cat_features=[1,2,3,4,5,6],
             #use_best_model=True,
             #early_stopping_rounds
             verbose=True)


    print_memory_info("after model.fit")
    print("==== BEFORE PICKLE ====")
    pickle.dump(model, open(model_filename, 'wb'))
    print("==== after PICKLE ====")
    return


import bcolz
import csv


cg1 = set()
cg2 = set()
cg3 = set()
cg1_nonzeros = 0 
cg2_nonzeros = 0
cg3_nonzeros = 0 


cg1_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451]

cg1_id2idx = {}
for i, cg1 in enumerate(cg1_list):
    cg1_id2idx[cg1] = i      

def main():
    # df = pd.read_csv('train.tar.gz', compression='gzip', header=0, sep=';', quotechar='"', nrows=10)
    # reader = pd.read_csv('train.tar.gz', compression='gzip', header=0, sep=';', quotechar='"', iterator=True, nrows=10)
    def generator_data():
        global cg1 
        global cg2 
        global cg3
        global cg1_nonzeros
        global cg2_nonzeros
        global cg3_nonzeros
        with open('train.csv', 'r') as f: 
            reader = csv.DictReader((line.replace('\0','') for line in f), delimiter=';') 
            for row in reader:
                # print(row)
                # cg1_  = [0]*len(cg1_list)
                for s in row["CG1"].split(','):
                    if s:
                        cg1_nonzeros += 1
                for s in row["CG2"].split(','):
                    if s:
                        cg2_nonzeros += 1
                for s in row["CG3"].split(','):
                    if s:
                        cg3_nonzeros += 1
                # cg1 |= set([int(s) for s in row["CG1"].split(',') if s])
                # cg2 |= set([int(s) for s in row["CG2"].split(',') if s])
                # cg3 |= set([int(s) for s in row["CG3"].split(',') if s])
                # if not row["CG2"]:
                #     row["CG2"]=np.nan
                # if not row["CG3"]:
                #     row["CG3"]=np.nan
                row_ = tuple([
                  row["timestamp"],
                  row["label"],
                    row["C1"],
                    row["C2"],
                    row["C3"],
                    row["C4"],
                    row["C5"],
                    row["C6"],
                    row["C7"],
                    row["C8"],
                    row["C9"],
                    # row["CG2"],
                    # row["CG3"],
                    row["l1"],
                    row["l2"],
                    row["C11"],
                    row["C12"]
                ])
                # print(row)
                #print(row_)
                yield row_
              
    # for row in generator_data():
    #     print(row)
                
    # ct = bcolz.ctable.fromdataframe(df, rootdir='train.bcolz')
    
    dtypes=[
                                        ("timestamp", 'i8'),
                                        ("label",'i2'),
                                        ('C1', 'i8'),
                                        ('C2', 'i8'),
                                        ('C3', 'i4'),
                                        ('C4', 'i4'),
                                        ('C5', 'i4'),
                                        ('C6', 'i4'),
                                        ('C7', 'i4'),
                                        ('C8', 'i4'),
                                        ('C9', 'i4'),
                                        # ('CG1', 'i4'),
                                        # ('CG2', 'f4'),
                                        # ('CG3', 'f4'),
                                        ('l1', 'i4'),
                                        ('l2', 'i4'),
                                        ('C11', 'i4'),
                                        ('C12', 'i4'),
    ]
    # for cg in cg1_list:
    #     dtypes.append( ("cg1_%03d" % cg, np.bool) )
    
    ct = bcolz.fromiter(generator_data(), dtype=dtypes, count=29989753, rootdir='train-v5.bcolz')
    print("cg1_nz=", cg1_nonzeros)
    print("cg2_nz=", cg2_nonzeros)
    print("cg3_nz=", cg3_nonzeros)
    pass  
                    
if __name__ == '__main__':
    # trainmodel()
    #analyse()
    main()
    pass



