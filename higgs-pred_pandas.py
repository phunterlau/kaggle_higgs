#!/usr/bin/python
# make prediction 
import sys
import numpy as np
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb
import pandas as pd
from add_features import add_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# path to where the data lies
dpath = 'data'

modelfile = 'higgs.model.%dstep.depth%s'%(steps_,sys.argv[2]))
# make top 15% as positive 
threshold_ratio = 0.14
outfile = 'higgs.pred.%.2f.%ssteps.depth%s.csv'%(threshold_ratio,sys.argv[1],sys.argv[2])

'''
df = pd.read_csv(dpath+'/training.csv')
#df.replace(-999.0,0.)

# rearrange columns for convenience
cols = df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df = df[cols]

df_new = add_features(df)
cols_new = df_new.columns.tolist()
cols_new = cols_new[:32]+cols_new[33:]+[cols_new[32]] #make the weight to the last
df_new=df_new[cols_new]
X_new = df_new[cols_new[2:-1]].values
#idx = dtest[:,0]
'''
df_test = pd.read_csv(dpath+'/test.csv')
#df_test.replace(-999.0,0.)
df_test_data = add_features(df_test)
cols_new = df_test_data.columns.tolist()
black_list = ['PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi',
              'PRI_tau_eta','PRI_lep_eta'] 
cols_new = [c for c in cols_new if c not in black_list]
df_test_data=df_test_data[cols_new]
data = df_test_data.values[:,1:]
#scaler = StandardScaler().fit(np.vstack((X_new, data))) #normalize with training scale
#scaler = MinMaxScaler(feature_range=(-10,10)).fit(np.vstack((X_new, data)))
#data = scaler.transform(data)
idx = df_test_data.values[:,0]

print ('finish loading from csv ')
xgmat = xgb.DMatrix( data, missing = -999.0 )
bst = xgb.Booster({'nthread':16})
bst.load_model( modelfile )
ypred = bst.predict( xgmat )

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ] 

rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1

# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:        
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'        
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()

print ('finished writing into prediction file')



