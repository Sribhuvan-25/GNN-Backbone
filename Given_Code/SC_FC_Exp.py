import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_sparse import spmm
import networkx as nx
import scipy.sparse as sp
from networkx.convert_matrix import from_numpy_matrix
import csv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, DummyExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.utils import to_networkx
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from dataset import SZ_SC_K_5_test, SZ_FC_K_5_test, SZ_SC_K_5_test_plus, SZ_FC_K_5_test_plus
from GNNmodels import simple_GCN_res, simple_GCN_res_plus, simple_RGGC, simple_RGGC_plus
from others import read_csv_folder, calculate_overall_average, save_to_csv, train, eval, train_plus, eval_plus, clear_folder



class MyGNNModel_SC_FC:
    def __init__(self, modality, learning_rate, dropout_value, batch_size,
                 channel_size, epoch_no, class_number, num_folds, GNN_dataset_path=None, 
                 Saved_model_path=None,Saved_feat_embedd_path=None,Loss_path=None,Saved_exp_path=None):
        self.modality = modality
        self.learning_rate = learning_rate
        self.dropout_value = dropout_value
        self.batch_size = batch_size
        self.channel_size = channel_size
        self.epoch_no = epoch_no
        self.class_number = class_number
        self.num_folds = num_folds
        self.GNN_dataset_path = GNN_dataset_path
        self.Saved_model_path = Saved_model_path
        self.Saved_feat_embedd_path = Saved_feat_embedd_path
        self.Loss_path = Loss_path
        self.Saved_exp_path = Saved_exp_path


    def SC_FC_exp_enh_model(self):

        if self.modality=='FC':
            dataset_full = SZ_FC_K_5_test(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5')
        elif self.modality=='SC':
            dataset_full = SZ_SC_K_5_test(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5')
        else:
            print(f'Select Modality (SC/FC) First !!')
        print(f'Number of subjects: {len(dataset_full)}')
        # dataset_full.filename

        HC_exp_path= f'{self.Saved_exp_path}/k_5_{self.modality}/HC'
        SZ_exp_path= f'{self.Saved_exp_path}/k_5_{self.modality}/SZ'
        All_exp_path= f'{self.Saved_exp_path}/k_5_{self.modality}/All'
        clear_folder(HC_exp_path)
        clear_folder(SZ_exp_path)
        clear_folder(All_exp_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        loss_fn_full = torch.nn.CrossEntropyLoss()

        all_test_acc = []
        all_train_acc = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_full)):
            train_dataset_fold = torch.utils.data.Subset(dataset_full, train_idx)
            test_dataset_fold = torch.utils.data.Subset(dataset_full, test_idx)
            
            train_loader_fold = DataLoader(train_dataset_fold, batch_size=self.batch_size, shuffle=True)
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=self.batch_size, shuffle=False)

            model_full = simple_RGGC(self.channel_size, self.class_number, self.dropout_value,input_channel=dataset_full.num_node_features).to(device)

            optimizer_full = torch.optim.Adam(model_full.parameters(), lr=self.learning_rate)
            
            train_losses = []
            test_acc = []
            train_acc = []
            test_pre= []
            test_rec= []
            test_f1= []
            validation_losses = []

            for epoch in range(self.epoch_no):
                loss = train(model_full, loss_fn_full, device, train_loader_fold, optimizer_full)
                validation_loss = 0.0

                model_full.eval()
                with torch.no_grad():
                    for batch in test_loader_fold:
                        batch = batch.to(device)
                        output = model_full(batch.x, batch.edge_index, batch.batch)
                        val_loss = loss_fn_full(output, batch.y)
                        validation_loss += val_loss.item()

                validation_loss /= len(test_loader_fold)
                validation_losses.append(validation_loss)

                train_result = eval(model_full, device, train_loader_fold)
                test_result = eval(model_full, device, test_loader_fold)

                train_losses.append(loss)
                test_acc.append(test_result)
                train_acc.append(train_result)

                y_true = []
                y_pred = []
                filenames = []
                with torch.no_grad():
                    for batch in test_loader_fold:
                        batch = batch.to(device)
                        output = model_full(batch.x, batch.edge_index, batch.batch)
                        y_true.extend(batch.y.cpu().numpy())
                        y_pred.extend(output.argmax(dim=1).cpu().numpy())
                        # Retrieve filenames
                        filenames.extend(batch.filename)

                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                test_pre.append(precision)
                test_rec.append(recall)
                test_f1.append(f1_score)

                print(f'Fold [{fold + 1}/{self.num_folds}], Epoch: {epoch + 1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_result:.2f}%, '
                    f'Test: {100 * test_result:.2f}%')

            all_test_acc.append(np.mean(test_acc))  
            all_train_acc.append(np.mean(train_acc)) 
            all_precisions.append(np.mean(test_pre)) 
            all_recalls.append(np.mean(test_rec)) 
            all_f1_scores.append(np.mean(test_f1)) 
            

            print(f'Average Train Accuracy for fold {fold + 1} is {100 * np.mean(train_acc):.2f}%')
            print(f'Average Test Accuracy for fold {fold + 1} is {100 * np.mean(test_acc):.2f}%')
            print(f'Average Test Precision for fold {fold + 1} is {100 * np.mean(test_pre):.2f}%')
            print(f'Average Test Recall for fold {fold + 1} is {100 * np.mean(test_rec):.2f}%')
            print(f'Average Test f1_score for fold {fold + 1} is {100 * np.mean(test_f1):.2f}%')
            # print("Filenames for test data in this fold:")
            # print(filenames)
                

            # Save the trained model
            model_filename = f'{self.Saved_model_path}/model_GCNConv_{self.modality}_k_5_fold_{fold+1}.pt'
            torch.save(model_full.state_dict(), model_filename)
            
            plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(validation_losses, label='Validation Loss')
            plt.title(f'Fold {fold + 1} Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{self.Loss_path}/GCNConv_{self.modality}_k_5_loss_fold_{fold + 1}.png')
            plt.close()
            
        avg_test_acc = np.mean(all_test_acc)
        avg_train_acc = np.mean(all_train_acc)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1_score = np.mean(all_f1_scores)
        print(f'Average Train Accuracy across {self.num_folds} folds: {100 * avg_train_acc:.2f}%')
        print(f'Average Test Accuracy across {self.num_folds} folds: {100 * avg_test_acc:.2f}%')
        print(f'Average Test Precision across {self.num_folds} folds: {100*avg_precision:.2f}%')
        print(f'Average Test Recall across {self.num_folds} folds: {100*avg_recall:.2f}%')
        print(f'Average Test F1 Score across {self.num_folds} folds: {100*avg_f1_score:.2f}%')


        models = []
        explainers = []

        for fold in range(1, 6):
            model_1 = model_full.to(device)
            state_dict = torch.load(f"{self.Saved_model_path}/model_GCNConv_{self.modality}_k_5_fold_{fold}.pt")
            model_1.load_state_dict(state_dict)
            models.append(model_1)

            gnnexplainer = Explainer(
                model=model_1,
                algorithm=GNNExplainer(epochs=300),
                explanation_type='model',
                node_mask_type=None,
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='probs',
                ),
                threshold_config=dict(
                    threshold_type='topk',
                    value=20
                )
            )
            explainers.append(gnnexplainer)

        def visualize_gnnexplainer_explanation(graph, graph_id, node_names=None, save_dir=None):
            """
            Visualize the explanation of a GNN using GNNExplainer.

            Parameters:
            - graph: GNN graph
            - graph_id: Identifier for the graph
            - node_names: List of node names (optional)
            - save_dir: Directory to save the edge importance matrix files

            Returns:
            None
            """
            graph = graph.to(device)
            model_full.to(device)
            prediction = model_full(graph.x, graph.edge_index, batch=torch.zeros(len(graph.x), dtype=int).to(device))
            predicted_label = torch.argmax(prediction).item()
            actual_label = graph.y.item()
            print(f"Actual label: {actual_label} vs Predicted label: {predicted_label}")

            explanation = gnnexplainer(graph.x, graph.edge_index, batch=torch.zeros(len(graph.x), dtype=int).to(device))
            graph = to_networkx(graph.cpu())
            graph = graph.copy().to_undirected()

            num_nodes = len(graph)
            edge_importance_matrix = np.zeros((num_nodes, num_nodes))

            for edge in graph.edges():
                u, v = edge
                edge_importance_matrix[u, v] = explanation.edge_mask[u].item()  # Store the importance score
                edge_importance_matrix[v, u] = explanation.edge_mask[u].item()  # Ensure symmetry

            All_subj_dir = f'{save_dir}/k_5_{self.modality}/All'
            save_path = os.path.join(All_subj_dir, f'edge_importance_matrix_{graph_id}.csv')
            np.savetxt(save_path, edge_importance_matrix, delimiter=',')    

            if actual_label == 1:
                SZ_dir = f'{save_dir}/k_5_{self.modality}/SZ'
                save_path_1 = os.path.join(SZ_dir, f'edge_importance_matrix_{graph_id}.csv')
                np.savetxt(save_path_1, edge_importance_matrix, delimiter=',')
            else:
                HC_dir = f'{save_dir}/k_5_{self.modality}/HC'
                save_path_1 = os.path.join(HC_dir, f'edge_importance_matrix_{graph_id}.csv')
                np.savetxt(save_path_1, edge_importance_matrix, delimiter=',')

        for i in range(len(test_dataset_fold)):
            graph = dataset_full[i]
            graph_id = i
            visualize_gnnexplainer_explanation(graph, graph_id, node_names=None, save_dir=self.Saved_exp_path)

        folder_path = f'{self.Saved_exp_path}/k_5_{self.modality}/All'
        data = read_csv_folder(folder_path)
        overall_average = calculate_overall_average(data)

        if overall_average is not None:
            output_path = f'{self.Saved_exp_path}/k_5_{self.modality}/All/average.csv'
            save_to_csv(overall_average, output_path)
            print(f"Overall average saved to {output_path}")
        else:
            print("No data found in the folder.")


        if self.modality=='FC':
            dataset_full_1 = SZ_FC_K_5_test_plus(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5_plus')
        elif self.modality=='SC':
            dataset_full_1 = SZ_SC_K_5_test_plus(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5_plus')
        else:
            print(f'Select Modality (SC/FC) First !!')
        print(f'Number of subjects: {len(dataset_full_1)}')
        # dataset_full_1.filename


        all_fine_tuned_test_acc = []
        all_fine_tuned_prec = []
        all_fine_tuned_f1_sc = []
        all_feat_embbeding = [] 
        all_target_values = []
        all_filenames = []  

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_full_1)):
            train_dataset_fold_1 = torch.utils.data.Subset(dataset_full_1, train_idx)
            test_dataset_fold_1 = torch.utils.data.Subset(dataset_full_1, test_idx)
            
            train_loader_full_1 = DataLoader(train_dataset_fold_1, batch_size=self.batch_size, shuffle=True)
            test_loader_fold_1 = DataLoader(test_dataset_fold_1, batch_size=self.batch_size, shuffle=False)

            model_path = f'{self.Saved_model_path}/model_GCNConv_{self.modality}_k_5_fold_{fold+1}.pt'
            model_2 = simple_RGGC_plus(self.channel_size, self.class_number, self.dropout_value, input_channel=dataset_full.num_node_features).to(device)
            model_2.load_state_dict(torch.load(model_path))

            optimizer = torch.optim.Adam(model_2.parameters(), lr=self.learning_rate)  

            fine_tuned_train_losses = []
            fine_tuned_test_acc = []
            fine_tuned_test_prec = []
            fine_tuned_test_f1_sc = []

            for epoch in range(self.epoch_no):  
                fine_tune_loss = train_plus(model_2, loss_fn_full, device, train_loader_full_1, optimizer)
                fine_tuned_train_losses.append(fine_tune_loss)

                test_acc = eval_plus(model_2, device, test_loader_fold_1) 
                fine_tuned_test_acc.append(test_acc)

                y_true_1 = []
                y_pred_1 = []
                with torch.no_grad():
                    for batch in test_loader_fold_1:
                        batch = batch.to(device)
                        output, _ = model_2(batch.x, batch.edge_index, batch.batch)
                        y_true_1.extend(batch.y.cpu().numpy())
                        y_pred_1.extend(output.argmax(dim=1).cpu().numpy())


                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true_1, y_pred_1, average='weighted', zero_division=0)
                fine_tuned_test_prec.append(precision)
                fine_tuned_test_f1_sc.append(f1_score)

                print(f'Fold [{fold + 1}/{self.num_folds}], Epoch: {epoch + 1:02d}, '
                    f'Fine-tune Loss: {fine_tune_loss:.4f}, '
                    f'Fine-tune Test Acc: {100 * test_acc:.2f}%,'
                    f'Fine-tune Test Pre: {100 * precision:.2f}%,'
                    f'Fine-tune Test F1_Sc: {100 * f1_score:.2f}%')
                

            all_fine_tuned_test_acc.append(np.mean(fine_tuned_test_acc))
            all_fine_tuned_prec.append(np.mean(fine_tuned_test_prec))
            all_fine_tuned_f1_sc.append(np.mean(fine_tuned_test_f1_sc))

            fine_tuned_model_path = f'{self.Saved_model_path}/fine_tuned_model_GCNConv_{self.modality}_k_5_fold_{fold+1}.pt'
            torch.save(model_2.state_dict(), fine_tuned_model_path)
            
            feat_embbeding = []
            target_values = []
            filenames = [] 
            model_2.eval()
            with torch.no_grad():
                for batch in test_loader_fold_1:
                    batch = batch.to(device)
                    _ , feat = model_2(batch.x, batch.edge_index, batch.batch)  
                    feat_embbeding.append(feat.cpu().numpy()) 
                    target_values.append(batch.y.cpu().numpy()) 
                    filenames.extend(batch.filename)   
            all_feat_embbeding.append(np.concatenate(feat_embbeding))  
            all_target_values.append(np.concatenate(target_values)) 
            all_filenames.append(filenames)  
            # print("Filenames for fold", fold + 1, ":", filenames)
            

        print(f'Average Train Accuracy across {self.num_folds} folds: {100 * np.mean(all_fine_tuned_test_acc):.2f}%\n'
            f'Average Train Precision across {self.num_folds} folds: {100 * np.mean(all_fine_tuned_prec):.2f}%\n'
            f'Average Train F1-Score across {self.num_folds} folds: {100 * np.mean(all_fine_tuned_f1_sc):.2f}% ')


        for fold, (feat_embbeding, target_values, filenames) in enumerate(zip(all_feat_embbeding, all_target_values, all_filenames)):
            np.save(f'{self.Saved_feat_embedd_path}/feat_embbeding_{self.modality}_k_5_fold_{fold+1}.npy', feat_embbeding)
            data = {'Filename': filenames, 'Target_Value': target_values}
            df = pd.DataFrame(data)
            df.to_csv(f'{self.Saved_feat_embedd_path}/target_values_{self.modality}_k_5_fold_{fold+1}.csv', index=False, header=False)


    def SC_FC_no_exp_enh_model(self):

        if self.modality=='FC':
            dataset_full = SZ_FC_K_5_test(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5')
        elif self.modality=='SC':
            dataset_full = SZ_SC_K_5_test(f'{self.GNN_dataset_path}/SZ_{self.modality}_K_5')
        else:
            print(f'Select Modality (SC/FC) First !!')
        print(f'Number of subjects: {len(dataset_full)}')
        # dataset_full.filename

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        loss_fn_full = torch.nn.CrossEntropyLoss()


        all_feat_embbeding = [] 
        all_target_values = []  
        all_test_acc = []
        all_train_acc = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_filenames = []  


        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_full)):
            train_dataset_fold = torch.utils.data.Subset(dataset_full, train_idx)
            test_dataset_fold = torch.utils.data.Subset(dataset_full, test_idx)
            
            train_loader_fold = DataLoader(train_dataset_fold, batch_size=self.batch_size, shuffle=True)
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=self.batch_size, shuffle=False)

            model_full = simple_RGGC_plus(self.channel_size, self.class_number, self.dropout_value,input_channel=dataset_full.num_node_features).to(device)

            optimizer_full = torch.optim.Adam(model_full.parameters(), lr=self.learning_rate)
            
            losses_full = []

            train_losses = []
            test_acc = []
            train_acc = []
            test_pre= []
            test_rec= []
            test_f1= []
            validation_losses = []

            

            for epoch in range(self.epoch_no):
                loss = train_plus(model_full, loss_fn_full, device, train_loader_fold, optimizer_full)
                train_result = eval_plus(model_full, device, train_loader_fold)
                test_result = eval_plus(model_full, device, test_loader_fold)

                losses_full.append(loss)
                test_acc.append(test_result)
                train_acc.append(train_result)

                y_true = []
                y_pred = []

                with torch.no_grad():
                    # val_loss = 0.0
                    for batch in test_loader_fold:
                        batch = batch.to(device)
                        output, _ = model_full(batch.x, batch.edge_index, batch.batch)
                        # val_loss += loss_fn_full(output, batch.y).item()
                        y_true.extend(batch.y.cpu().numpy())
                        y_pred.extend(output.argmax(dim=1).cpu().numpy())
                    # avg_val_loss = val_loss / len(test_loader_fold)
                    # validation_losses.append(avg_val_loss)

                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                test_pre.append(precision)
                test_rec.append(recall)
                test_f1.append(f1_score)

                print(f'Fold [{fold + 1}/{self.num_folds}], Epoch: {epoch + 1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_result:.2f}%, '
                    f'Test: {100 * test_result:.2f}%')

            all_test_acc.append(np.mean(test_acc))  
            all_train_acc.append(np.mean(train_acc)) 
            all_precisions.append(np.mean(test_pre)) 
            all_recalls.append(np.mean(test_rec)) 
            all_f1_scores.append(np.mean(test_f1)) 

            

            print(f'Average Train Accuracy for fold {fold + 1} is {100 * np.mean(train_acc):.2f}%')
            print(f'Average Test Accuracy for fold {fold + 1} is {100 * np.mean(test_acc):.2f}%')
            print(f'Average Test Precision for fold {fold + 1} is {100 * np.mean(test_pre):.2f}%')
            print(f'Average Test Recall for fold {fold + 1} is {100 * np.mean(test_rec):.2f}%')
            print(f'Average Test f1_score for fold {fold + 1} is {100 * np.mean(test_f1):.2f}%')
            

            feat_embbeding = []
            target_values = []
            filenames = []
            model_full.eval()
            with torch.no_grad():
                for batch in test_loader_fold:
                    batch = batch.to(device)
                    _ , xx = model_full(batch.x, batch.edge_index, batch.batch)  
                    feat_embbeding.append(xx.cpu().numpy()) 
                    target_values.append(batch.y.cpu().numpy()) 
                    filenames.extend(batch.filename) 
            all_feat_embbeding.append(np.concatenate(feat_embbeding))  
            all_target_values.append(np.concatenate(target_values)) 
            all_filenames.append(filenames)  

            model_filename = f'{self.Saved_model_path}/model_GCNConv_{self.modality}_k_5_no_exp_fold_{fold+1}.pt'
            torch.save(model_full.state_dict(), model_filename)
 

        avg_test_acc = np.mean(all_test_acc)
        avg_train_acc = np.mean(all_train_acc)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1_score = np.mean(all_f1_scores)
        print(f'Average Train Accuracy across {self.num_folds} folds: {100 * avg_train_acc:.2f}%')
        print(f'Average Test Accuracy across {self.num_folds} folds: {100 * avg_test_acc:.2f}%')
        print(f'Average Test Precision across {self.num_folds} folds: {100*avg_precision:.2f}%')
        print(f'Average Test Recall across {self.num_folds} folds: {100*avg_recall:.2f}%')
        print(f'Average Test F1 Score across {self.num_folds} folds: {100*avg_f1_score:.2f}%')


        for fold, (feat_embbeding, target_values, filenames) in enumerate(zip(all_feat_embbeding, all_target_values, all_filenames)):
            np.save(f'{self.Saved_feat_embedd_path}/feat_embbeding_{self.modality}_k_5_no_exp_fold_{fold+1}.npy', feat_embbeding)
            data = {'Filename': filenames, 'Target_Value': target_values}
            df = pd.DataFrame(data)
            df.to_csv(f'{self.Saved_feat_embedd_path}/target_values_{self.modality}_k_5_no_exp_fold_{fold+1}.csv', index=False, header=False)