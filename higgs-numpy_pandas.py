#!/usr/bin/python
# this is the example script to use xgboost to train 
import inspect
import os
import sys
import numpy as np
import pandas as pd
from add_features import add_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")

sys.path.append(code_path)

import xgboost as xgb

test_size = 550000

steps_ = int(sys.argv[1])

# path to where the data lies
dpath = 'data'

def get_training_data(training_file, test_file):
    '''
    Loads training data.
    '''
    # load training data
    df = pd.read_csv(training_file)

    # map y values to integers
    df['Label'] = df['Label'].map({'b':0, 's':1})

    # rearrange columns for convenience
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]

    print 'original features'
    print df.columns

    df_new = add_features(df)
    #df_new = df
    cols_new = df_new.columns.tolist()
    cols_new = cols_new[:32]+cols_new[33:]+[cols_new[32]] #make the weight to the last
    #the ending comma!!
    black_list = ['PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi',
                  'PRI_tau_eta','PRI_lep_eta',
                  'PRI_jet_leading_eta','PRI_jet_subleading_eta',#replace with abs values
                  'PRI_lep_px','PRI_lep_py','PRI_lep_pz', 'PRI_lep_px_abs','PRI_lep_py_abs',#these raw values are noisy
                  'PRI_tau_px','PRI_tau_py','PRI_tau_pz', 'PRI_tau_pz_abs', 
                  'PRI_jet_leading_px','PRI_jet_leading_py','PRI_jet_leading_pz', #leading pxyz has separation but abs
                  'PRI_jet_subleading_px','PRI_jet_subleading_py','PRI_jet_subleading_pz',
                  ] 
    #experiment if these phi values makes no sense: TRUE phi itself is really noisy as expected
    cols_new = [c for c in cols_new if c not in black_list]
    df_new=df_new[cols_new]
    print 'newly added features'
    print df_new.columns
    # convert into numpy array
    #train_data = df_new.values
    print 'select X features ', cols_new[2:-1]
    X_new = df_new[cols_new[2:-1]].values
    labels = df_new['Label'].values
    weights = df_new['Weight'].values

    #print 'exporting to csv with additional feat'
    #df_new.to_csv('./additional_feat_training.csv')
    #sys.exit()
    return X_new, labels, weights

'''
    #load test data for better scaling
    df_test = pd.read_csv(test_file)
    df_test.replace(-999.0,0.)
    df_test = df_test[df_test['DER_mass_MMC']>-999.0]
    df_test_data = add_features(df_test)
    X_test = df_test_data.values[:,2:]
    #scaler = StandardScaler().fit(np.vstack((X_new, X_test)))
    scaler = MinMaxScaler(feature_range=(-10,10)).fit(np.vstack((X_new, X_test)))
    #scaler = StandardScaler().fit(X_new)
    X_new = scaler.transform(X_new)
'''

# load in training data, directly use numpy
#dtrain = np.loadtxt( dpath+'/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
print ('finish loading from csv ')

data, label, weight = get_training_data(dpath+'/training.csv', dpath+'/test.csv')
#label  = dtrain[:,32]
#data   = dtrain[:,1:31]
# rescale weight to make it same as test set
weight = weight * float(test_size) / len(label)

sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

# print weight statistics 
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )

# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
#param['objective'] = 'binary:logistic'
# scale weight of positive examples
#e 0.07 s 0.95 m 8 round 120
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['bst:eta'] = 0.01
#param['bst:max_depth'] = 9
param['bst:max_depth'] = int(sys.argv[2])
param['bst:subsample'] = 0.9
param['eval_metric'] = 'ams@0.14'
#param['eval_metric'] = 'error'
param['silent'] = 1
param['nthread'] = 16

# you can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items())#+[('eval_metric', 'ams@0.15')]

watchlist = [ (xgmat,'train') ]
# boost 120 tres
num_round = steps_
#num_round = 200
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist );
# save out model
bst.save_model('higgs.model.%dstep.depth%s'%(steps_,sys.argv[2]))

print ('finish training')
