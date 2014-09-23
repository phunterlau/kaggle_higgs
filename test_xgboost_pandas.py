import csv
import sys
sys.path.append('../../python/')
import numpy as np
import scipy as sp
import xgboost as xgb
import sklearn.cross_validation as cv
import itertools
import pandas as pd
from add_features import add_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier

def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_rates(prediction, solution, weights):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size
    assert prediction.size == weights.size

    # Compute sum of weights for true and false positives
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    return truePos, falsePos


def get_training_data(training_file, test_file):
    '''
    Loads training data.
    '''
    # load training data
    df = pd.read_csv(training_file)
    #df.replace(-999.0,0.)

    # map y values to integers
    df['Label'] = df['Label'].map({'b':0, 's':1})

    # rearrange columns for convenience
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]

    #print df.columns

    df_new = add_features(df)
    cols_new = df_new.columns.tolist()
    cols_new = cols_new[:32]+cols_new[33:]+[cols_new[32]] #make the weight to the last
    #print len(cols_new)
    #black_list = ['PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi',
    #              'PRI_tau_eta','PRI_lep_eta'
    #              ] 
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
    #print df_new.columns
    X_new = df_new[cols_new[2:-1]].values
    labels = df_new['Label'].values
    weights = df_new['Weight'].values

    #load test data for better scaling
    #df_test = pd.read_csv(test_file)
    #df_test.replace(-999.0,0.)
    #df_test_data = add_features(df_test)
    #X_test = df_test_data.values[:,1:]

    #tree-based feature selection
    #treeClf = ExtraTreesClassifier()
    #print 'feature selection'
    #treeClf = AdaBoostClassifier(n_estimators=1000,learning_rate=0.1)
    #X_new = treeClf.fit(X, labels).transform(X)
    #print treeClf.feature_importances_

    #print X_new.shape

    #scaler = StandardScaler().fit(X_new)
    #standardize the training along with the test scale
    #scaler = StandardScaler().fit(np.vstack((X_new, X_test)))
    #scaler = MinMaxScaler(feature_range=(-10,10)).fit(np.vstack((X_new, X_test)))
    #scaler = Binarizer().fit(np.vstack((X_new, X_test)))
    #X_new = scaler.transform(X_new)

    return X_new, labels, weights


def estimate_performance_xgboost(training_file, test_file, param, num_round, folds):
    '''
    Cross validation for XGBoost performance 
    '''
    # Load training data
    X, labels, weights = get_training_data(training_file, test_file)

    # Cross validate
    kf = cv.KFold(labels.size, n_folds=folds)
    npoints  = 26
    # Dictionary to store all the AMSs
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    # These are the cutoffs used for the XGBoost predictions
    cutoffs  = sp.linspace(0.05, 0.30, npoints)
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]

        # Rescale weights so that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        #param['objective'] = 'binary:logitraw'
        #param['objective'] = 'binary:logistic'
        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg/sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here 
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        #watchlist = [(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        # Construct matrix for test set
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        best_thres = 0.0
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = sp.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_test, w_test)
            this_AMS = AMS(truePos, falsePos)
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
                best_thres = threshold_ratio
        print "Best AMS = %f at %.2f"%(best_AMS,best_thres)
    print "------------------------------------------------------"
    for curr, cut in enumerate(cutoffs):
        print "Thresh = %.2f: AMS = %.4f, std = %.4f" % \
            (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr]))
    print "------------------------------------------------------"


def main():
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logitraw'
    #param['bst:eta'] = 0.2
    #param['bst:max_depth'] = 6
    #param['bst:subsample'] = 0.3
    #param['eval_metric'] = 'auc'
    param['eval_metric'] = 'ams@0.14'
    param['silent'] = 1
    param['nthread'] = 16

    num_round = 120 # Number of boosted trees
    folds = 5 # Folds for CV
    #estimate_performance_xgboost("./data/training.csv", param, num_round, folds)

    all_etas = [0.01]
    all_subsamples = [0.7,0.9]
    all_depth = [9,10]
    nums_rounds = [3000]
    #all_etas = [0.01]
    #all_subsamples = [0.9,0.95,0.85]
    #all_depth = [8,9,10]
    #nums_rounds = [3000]
    e_s_m = list(itertools.product(all_etas,all_subsamples,all_depth,nums_rounds))
    for e,s,m,r in e_s_m:
        param['bst:eta'] = e
        param['bst:subsample'] = s
        param['bst:max_depth'] = m
        print 'e %.3f s %.2f m %d round %d'%(e,s,m,r)
        estimate_performance_xgboost("./data/training.csv", "./data/test.csv", param, r, folds)


if __name__ == "__main__":
    main()
