kaggle_higgs
============

My winning solution for Kaggle Higgs Machine Learning Challenge (single classifier, xgboost)

The full description is linked to this page:
http://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/

Thanks to xgboost for providing the fast GBM model.
Thanks to glh3 for their original work on feature interface https://github.com/ghl3/higgs-kaggle 

#usage

test_xgboost_pandas.py: the cross validation script. the original version is from Kaggle forum https://www.kaggle.com/c/higgs-boson/forums/t/8207/to-ams-3-6-model-can-you-share-you-local-cv-score/44825#post44825

higgs-numpy_pandas.py: the model generating script

higgs-pred_pandas.py: the submission generating script

run.all.sh: a simple script of generating model and generating submission

By running the default parameter of 3000 steps and max_depth of 9, one can reproduce the current best leaderboard score of 3.73 which is ranked 25th/1792
