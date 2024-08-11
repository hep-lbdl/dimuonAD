import numpy as np
import torch

from scipy.stats import wasserstein_distance

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score



def sample_from_flow(model_path, device, context, num_features):
    
    print(f"Loading in the best flow model ...")
    flow_best = torch.load(model_path)
    flow_best.to(device)

    # freeze the trained model
    for param in flow_best.parameters():
        param.requires_grad = False
    flow_best.eval()

    context_masses = torch.tensor(context.reshape(-1,1)).float().to(device)
    SB_samples = flow_best.sample(1, context=context_masses).detach().cpu().numpy()
    SB_samples = SB_samples.reshape(SB_samples.shape[0], num_features)

    SB_samples = np.hstack((SB_samples, np.reshape(context, (-1, 1))))
    
    return SB_samples


def get_1d_wasserstein_distances(samples_1, samples_2):
    
    distances_1d = []
    for i in range(samples_1.shape[1]):
        distances_1d.append(wasserstein_distance(samples_1[:,i] , samples_2[:,i]))
    return distances_1d


def run_BDTs(train_samp_0, train_samp_1, test_samp_0, test_samp_1, num_to_ensemble):
    
    X_train = np.vstack((train_samp_0, train_samp_1))
    Y_train = np.vstack((np.zeros((train_samp_0.shape[0], 1)), np.ones((train_samp_1.shape[0], 1))))

    X_test = np.vstack((test_samp_0, test_samp_1))
    Y_test = np.vstack((np.zeros((test_samp_0.shape[0], 1)), np.ones((test_samp_1.shape[0], 1))))
    
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    class_weight = {0: 1, 1: sum(Y_train==0)[0]/sum(Y_train==1)[0]}
    class_weights = class_weight[0]*(1.0-Y_train)+class_weight[1]*Y_train
    Y_train = Y_train.reshape(-1,)
    Y_test = Y_test.reshape(-1,)
    class_weights = class_weights.reshape(-1,)

    print("\nTraining class weights: ", class_weight)

    scores = {}

    for i in range(num_to_ensemble):
        
        print("Tree number:", i)
        np.random.seed(i+1)
        
        tree = HistGradientBoostingClassifier(verbose=0, max_iter=200, max_leaf_nodes=31, validation_fraction=0.5)
        results_f = tree.fit(X_train, Y_train, sample_weight=class_weights)
        loc_scores = tree.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(Y_test, loc_scores)
        scores[i] = {"fpr":fpr, "tpr": tpr}
        
        del results_f
        
    return scores

def run_NNs(train_samp_0, train_samp_1, test_samp_0, test_samp_1, num_to_ensemble, device):
    
    hyperparameters_dict_BC = {"n_epochs":50,
                              "batch_size": 512,
                              "lr": 0.001,
                             }

    scores = {}
    Y_test = np.vstack((np.zeros((test_samp_0.shape[0], 1)), np.ones((test_samp_1.shape[0], 1))))
    
    for i in range(num_to_ensemble):
        auc, fpr, tpr, outputs = discriminate_datasets_kfold("tmp", train_samp_0,  train_samp_1, 
                                np.ones((train_samp_0.shape[0],1)), np.ones((train_samp_1.shape[0],1)), 
                                   test_samp_0, test_samp_1, train_samp_0.shape[1], 
                                hyperparameters_dict_BC, device, seed = i+1, visualize = False, k_folds = 2)
        
        fpr, tpr, _ = roc_curve(Y_test, outputs)
        scores[i] = {"fpr":fpr, "tpr": tpr}

    return scores