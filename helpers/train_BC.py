import numpy as np
from helpers.make_BC import *

from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import roc_auc_score, roc_curve


def discriminate_datasets_kfold(results_dir, train_samp_1, train_samp_2, weights_samp_1, weights_samp_2, test_samp_1, test_samp_2, n_features, hyperparameters_dict, device, early_stop = True, visualize = True, seed = 2515, k_folds = 5):
    
    n_epochs = hyperparameters_dict["n_epochs"]
    batch_size = hyperparameters_dict["batch_size"]
    lr = hyperparameters_dict["lr"]
    
    if seed is not None:
        print(f"Using seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    X_train = np.concatenate((train_samp_1, train_samp_2))
    y_train = np.concatenate((torch.zeros((train_samp_1.shape[0], 1)), torch.ones((train_samp_2.shape[0],1))))    
    w_train = np.concatenate((weights_samp_1, weights_samp_2))
    
    # get weights in case we're oversampling
    class_weights = class_weight.compute_class_weight(class_weight ='balanced', 
                                                      classes = np.unique(y_train.reshape(-1)),
                                                      y = y_train.reshape(-1))
    class_weights = dict(enumerate(class_weights))   
    
    X_test = np.concatenate((test_samp_1, test_samp_2))
    y_test = np.concatenate((torch.zeros((test_samp_1.shape[0], 1)), torch.ones((test_samp_2.shape[0],1))))
    
    print("Train data, labels shape:", X_train.shape, y_train.shape)
    print("Test data, labels  shape:", X_test.shape, y_test.shape)
    
    # send to device
    X_train = np_to_torch(X_train, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    w_train = np_to_torch(w_train, device)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_best_val_losses = []
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):     
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        X_train_fold = X_train[train_ids]
        y_train_fold = y_train[train_ids]
        w_train_fold = w_train[train_ids] 
        
        X_val_fold = X_train[val_ids]
        y_val_fold = y_train[val_ids]
        w_val_fold = w_train[val_ids] 
                
        train_set = torch.utils.data.TensorDataset(X_train_fold, y_train_fold, w_train_fold)
        val_set = torch.utils.data.TensorDataset(X_val_fold, y_val_fold, w_val_fold)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
        
        # initialze the network
        dense_net = NeuralNet(input_shape = n_features)
        criterion = F.binary_cross_entropy 
        optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)
        dense_net.to(device)
        
        if early_stop:
            early_stopping = EarlyStopping()
        
         # save the best model
        val_loss_to_beat = 1e10
        best_epoch = -1

        epochs, losses, losses_val = [], [], []

        for epoch in tqdm(range(n_epochs)):
            losses_batch_per_e = []
            # batching    
            for batch_index, (batch_data, batch_labels, batch_salad_weights) in enumerate(train_loader):

                # calculate the loss, backpropagate
                optimizer.zero_grad()

                # get the weights
                batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                            - batch_labels)*class_weights[0] \
                            + batch_labels*class_weights[1]
                batch_weights = batch_weights_class*batch_salad_weights
                         
                loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses_batch_per_e.append(loss.detach().cpu().numpy())

            epochs.append(epoch)
            losses.append(np.mean(losses_batch_per_e))

            # validation
            with torch.no_grad():
                val_losses_batch_per_e = []
                
                for batch_index, (batch_data, batch_labels, batch_salad_weights) in enumerate(val_loader):
                    # calculate the loss, backpropagate
                    optimizer.zero_grad()

                    # get the weights
                    batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]
                    
                    batch_weights = batch_weights_class*batch_salad_weights

                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

                losses_val.append(np.mean(val_losses_batch_per_e))

                # see if the model has the best val loss
                if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                    val_loss_to_beat = np.mean(val_losses_batch_per_e)
                    # save the modelnp.rando
                    model_path = f"{results_dir}/.bc_fold{fold}.pt"
                    torch.save(dense_net, model_path)
                    best_epoch = epoch

                if early_stop:
                    early_stopping(np.mean(val_losses_batch_per_e))

            if early_stopping.early_stop:
                break
        
        print(f"Done training fold {fold}. Best val loss {val_loss_to_beat} at epoch {best_epoch}")

        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax.plot(epochs, losses)
            ax.plot(epochs, losses_val, label = "val")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"fold{fold}")

        # evaluate
        fold_best_val_losses.append(val_loss_to_beat)
            
    
    # load in the model / fold with the best val loss 
    best_model_index = np.argmin(fold_best_val_losses)
    best_model_path = f"{results_dir}/.bc_fold{best_model_index}.pt"
    print(f"Loading in best model for {best_model_path}, val loss {np.min(fold_best_val_losses)} from fold {best_model_index}")
    
    dense_net_eval = torch.load(best_model_path)
    dense_net_eval.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net_eval(X_test).detach().cpu().numpy()
        predicted = np.round(outputs)

        # calculate auc 
        auc = roc_auc_score(y_test, outputs)
        fpr, tpr, _ = roc_curve(y_test, outputs)

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC: " + str(auc))
        
    np.save(f"{results_dir}/fpr", fpr)
    np.save(f"{results_dir}/tpr", tpr)
        
    if auc < 0.5:
        auc = 1.0 - auc
    
    return auc, outputs

