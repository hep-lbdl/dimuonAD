import numpy as np
import xgboost as xgb


def custom_logloss_for_optim(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    
    preds = 1.0/(1 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad*weights, hess*weights


def custom_logloss_for_metric(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    preds = 1.0/(1.0 + np.exp(-preds))
    
    logloss = -(labels*np.log(preds) + (1.0-labels)*(np.log(1.0-preds)))
    return "logloss_mine", np.mean(weights*logloss)
    
    
def custom_predict(booster, dtest):
    
    bdt_outputs = booster.predict(dtest)
    return 1.0 / (1.0+np.exp(-bdt_outputs))

