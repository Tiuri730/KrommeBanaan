#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:28:57 2018

@author: york
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

#Load data
def load_data():
    X = pd.read_csv('train_X2.csv', header = 0, index_col = 0)
    y = pd.read_csv('train_y2.csv', header = 0, index_col = 0)
    test = pd.read_csv('test_extended2.csv', header = 0, index_col = 0)
    return X, y, test

#Stratified K-fold cross validation (using ROC AUC), with stability selection
def crossVal(X, y, n, params):
    skf = StratifiedKFold(n_splits=n, shuffle = True)
    test_scores = {}
    
    iter = 0
    
    for train_i, test_i in skf.split(X,y):
        print("Beginning stability selection: %i" %iter)
        X_train, y_train = X.iloc[train_i],y.iloc[train_i]
        X_test, y_test = X.iloc[test_i], y.iloc[test_i]
        
        #Create XGB model
        xgbclf = xgb.XGBClassifier(objective = 'binary:logistic', silent = True, nthread = 1)
        
        #Parameters for model fit
        fit_params = {'eval_metric': 'auc',
                      'verbose': False,
                      'early_stopping_rounds': 30,
                      'eval_set': [(X_test, y_test)]
                      }

        #Stability selection
        skf2 = StratifiedKFold(n_splits=2, shuffle = True)
        grid_search = GridSearchCV(xgbclf, param_grid=params, cv=skf2, refit = True, scoring = 'roc_auc', verbose = 30, n_jobs = -2)
#        grid_search = RandomizedSearchCV(xgbclf, param_distributions=params, n_iter=50, cv=skf2, refit = True, scoring = 'roc_auc', verbose = 10, n_jobs = -2)
        grid_search.fit(X_train, y_train, **fit_params) #Use early stopping during fitting of the model
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('newtest2_cv_results_fold_' + str(iter) + '.csv')
#        #Train model with the best settings
        best_model = grid_search.best_estimator_
#        xgb.plot_importance(best_model, importance_type = 'gain')
        #Predict on held-out test set with best model
        y_hat = best_model.predict_proba(X_test)[:,1]
        test_scores[iter] = roc_auc_score(y_test, y_hat)
#        val_scores['fold_'+str(iter)+'_stats'] = grid_search.cv_results_['mean_test_score']
#        val_params['fold_'+str(iter)+'_params'] = grid_search.cv_results_['params']
#        test_scores['fold_'+str(iter) + '_best_test'] = roc_auc_score(y_test, y_hat)
#        best_params['fold_'+str(iter) + '_best_params'] = grid_search.best_params_
        iter += 1
#    return test_scores, val_scores, best_params, val_params
    return results, test_scores




if __name__ == "__main__": #Protect for parallel processing
    #Get data
    X, y, test = load_data()
    
    #Setup a parameter grid to search
#    param_grid = {
#                    'max_depth': [2, 3, 5],
#                    'learning_rate': [0.01, 0.05, 0.1],
#                    'n_estimators': [100, 200, 400, 600],
#                    'min_child_weight': [1, 5, 10],
#                    'subsample': [0.6, 0.8, 1.0],
#                    'gamma': [0.5, 1, 2, 5]
#                    }
    
    param_grid = {
                'max_depth': [10, 20],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [300,600,900],
                'min_child_weight': [5, 10],
                'subsample': [0.8, 1.0],
                }
    
#    Test grid
#    param_grid = {"max_depth": [3],
#              "learning_rate": [0.1],
#              "n_estimators": [10]}
    
    #Cross validation method
    results, test_scores = crossVal(X, y['is_delayed'], 5, param_grid)
    #test_scores, val_scores, best_params, val_params = crossVal(X, y['is_delayed'], 5, param_grid)
    #cross_val_scores_smote = crossVal_smote(xgb_model, X, y, 10)
    #best_model, cv_results, best_params = crossVal2(X,y['is_delayed'],5,param_grid)
#    test_scores = pd.DataFrame(list(test_scores.items()))
#    val_scores = pd.DataFrame(list(val_scores.item()))
#    best_params = pd.DataFrame(list(best_params.items()))
#    val_params = pd.DataFrame(list(val_params.items()))
#    
#    test_scores.to_csv('results/test_scores.csv')
#    val_scores.to_csv('results/val_scores.csv')
#    best_params.to_csv('results/best_params.csv')
#    val_params.to_csv('results/val_params.csv')
#    np.savetxt("best_params.csv", best_params, delimiter=",", fmt='%s')
    
    
#    csv_list = ['test_scores.csv','val_scores.csv','best_params.csv', 'val_params.csv']
#    for f in csv_list:
#        with open(f, 'w') as myfile:
#            for k,v in best_params.items():
#                myfile.write(str(k) + ' : ' + str(v) + '\n\n')
    
	
    #Prepare results for submission
#    y_hat = best_model.predict_proba(test)
#    y_hat = pd.DataFrame(y_hat).drop(0, axis = 1) #Drop first column, as this predicts when flight is NOT delayed, we are only interested in flights that are delayed
#    y_hat['id']= test.index #Add the test id's
#    y_hat = y_hat.set_index('id') #Set index to be the id's
#    y_hat = y_hat.rename(columns = {1 : 'is_delayed'}) #Rename the only column to 'is_delayed'
#    y_hat.to_csv('results_jsn235_jor250.csv')