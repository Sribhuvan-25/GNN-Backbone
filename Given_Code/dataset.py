import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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



dataset_path = './FBIRN_SC_FC'
corr_matrices_dir = f'{dataset_path}/sFNC'
SC_matrices_dir = f'{dataset_path}/DTI_SC'
labels_file = f'{dataset_path}/FBIRN_SCFC_label.csv' 
SNP_dir = f'{dataset_path}/snp'
k_value = 5

saved_exp_path= './explanations/GCN'

corr_file_paths = sorted(os.listdir(corr_matrices_dir))
SC_file_paths = sorted(os.listdir(SC_matrices_dir))
snp_file_paths = sorted(os.listdir(SNP_dir))
matching_filenames = set(corr_file_paths) & set(SC_file_paths) & set(snp_file_paths)
sorted_matching_filenames = sorted(matching_filenames, reverse=True)


def local_degree_profiles(adjacency_matrix, radius):
    adj_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    ldp_list = []
    num_nodes = adj_matrix_tensor.shape[0]
    identity = torch.eye(num_nodes)
    
    node_degrees = []
    
    for node in range(num_nodes):
        node_neighbors = identity[node].clone()
        ldp = []
        
        # Calculate node degree
        node_degree = torch.sum(adj_matrix_tensor[node]).item()
        node_degrees.append(node_degree)
        
        for _ in range(radius):
            node_neighbors = torch.mm(adj_matrix_tensor, node_neighbors.unsqueeze(1)).squeeze(1)
            ldp.append(torch.sum(node_neighbors).item())
        
        ldp_list.append(ldp)
    mean_ldp = [torch.mean(torch.tensor(ldp)).item() for ldp in ldp_list]
    max_ldp = [torch.max(torch.tensor(ldp)).item() for ldp in ldp_list]
    min_ldp = [torch.min(torch.tensor(ldp)).item() for ldp in ldp_list]
    std_ldp = [torch.std(torch.tensor(ldp)).item() for ldp in ldp_list]
    combine= np.concatenate((node_degrees, mean_ldp,max_ldp,min_ldp,std_ldp))
    reshaped_combine = np.reshape(combine, (num_nodes, 5))
    # print(reshaped_combine)
    return reshaped_combine


class SZ_SC_K_5_test(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, neighbors=k_value):
  # def __init__(self, root, transform=None, pre_transform=None):
    self.neighbors = neighbors
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_file_names(self):
    return ['data.pt']

  def process(self):
    graphs = []
    filename_column = 0
    data_column = 1

    for i, filename in enumerate(sorted_matching_filenames):

      corr_matrices_dir_path = os.path.join(corr_matrices_dir, filename)
      SC_matrices_dir_path = os.path.join(SC_matrices_dir, filename)
      print(f"Processing files: {corr_matrices_dir_path} and {SC_matrices_dir_path}")
      print(f"Processing files: {corr_matrices_dir_path}")
      
      ts = np.loadtxt(SC_matrices_dir_path, delimiter=',')
      avg_pcorr_matrix_ts = np.loadtxt(SC_matrices_dir_path, delimiter=',')
      
      index = avg_pcorr_matrix_ts.argsort(axis=1)
      n_rois = avg_pcorr_matrix_ts.shape[0]
    
      for i in range(n_rois):
        for j in range(n_rois - self.neighbors):
          avg_pcorr_matrix_ts[i, index[i, j]] = 0
        for j in range(n_rois - self.neighbors, n_rois):
          avg_pcorr_matrix_ts[i, index[i, j]] = 1  

      SC_matrix_nx = from_numpy_matrix(avg_pcorr_matrix_ts)         
      pcorr_matrix_data = from_networkx(SC_matrix_nx)     
      filename_without_extension = os.path.splitext(filename)[0]
        
      with open(labels_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
            
        for row in csv_reader:
          if filename_without_extension == row[filename_column]:
            ii=row[data_column]
            print(f"Filename: {filename}, Data: {ii}") 

      lower_triangle = np.tril(ts)
      combine_features = local_degree_profiles(lower_triangle, 2)
      pcorr_matrix_data.x = torch.tensor(combine_features).float()   
      pcorr_matrix_data.y = torch.tensor(int(ii)) 
      pcorr_matrix_data.filename = filename_without_extension
      graphs.append(pcorr_matrix_data)

    data, slices = self.collate(graphs)
    torch.save((data, slices), self.processed_paths[0])


class SZ_FC_K_5_test(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, neighbors=k_value):
  # def __init__(self, root, transform=None, pre_transform=None):
    self.neighbors = neighbors
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_file_names(self):
    return ['data.pt']

  def process(self):
    graphs = []
    filename_column = 0
    data_column = 1


    for i, filename in enumerate(sorted_matching_filenames):

      corr_matrices_dir_path = os.path.join(corr_matrices_dir, filename)
      SC_matrices_dir_path = os.path.join(SC_matrices_dir, filename)
      print(f"Processing files: {corr_matrices_dir_path} and {SC_matrices_dir_path}")
      print(f"Processing files: {corr_matrices_dir_path}")
      
      ts = np.loadtxt(corr_matrices_dir_path, delimiter=',')
      avg_pcorr_matrix_ts = np.loadtxt(corr_matrices_dir_path, delimiter=',')
      
      index = avg_pcorr_matrix_ts.argsort(axis=1)
      n_rois = avg_pcorr_matrix_ts.shape[0]
    
      for i in range(n_rois):
        for j in range(n_rois - self.neighbors):
          avg_pcorr_matrix_ts[i, index[i, j]] = 0
        for j in range(n_rois - self.neighbors, n_rois):
          avg_pcorr_matrix_ts[i, index[i, j]] = 1  

      SC_matrix_nx = from_numpy_matrix(avg_pcorr_matrix_ts)         
      pcorr_matrix_data = from_networkx(SC_matrix_nx)     
      filename_without_extension = os.path.splitext(filename)[0]
        
      with open(labels_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
            
        for row in csv_reader:
          if filename_without_extension == row[filename_column]:
            ii=row[data_column]
            print(f"Filename: {filename}, Data: {ii}") 

      lower_triangle = np.tril(ts)
      combine_features = local_degree_profiles(lower_triangle, 2)
      pcorr_matrix_data.x = torch.tensor(combine_features).float()   
      pcorr_matrix_data.y = torch.tensor(int(ii)) 
      pcorr_matrix_data.filename = filename_without_extension
      graphs.append(pcorr_matrix_data)

    data, slices = self.collate(graphs)
    torch.save((data, slices), self.processed_paths[0])



class SZ_SC_K_5_test_plus(InMemoryDataset):
#   def __init__(self, root, transform=None, pre_transform=None, neighbors=5):
  def __init__(self, root, transform=None, pre_transform=None):
    # self.neighbors = neighbors
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_file_names(self):
    return ['data.pt']

  def process(self):
    graphs = []
    filename_column = 0
    data_column = 1

    SC_matrices_dir_path_1= f'{saved_exp_path}/k_5_SC/All/average.csv'

    for i, filename in enumerate(sorted_matching_filenames):

      corr_matrices_dir_path = os.path.join(corr_matrices_dir, filename)
      SC_matrices_dir_path = os.path.join(SC_matrices_dir, filename)
      print(f"Processing files: {corr_matrices_dir_path} and {SC_matrices_dir_path}")
      print(f"Processing files: {corr_matrices_dir_path}")
      
      ts = np.loadtxt(SC_matrices_dir_path, delimiter=',')
      avg_pcorr_matrix_ts = np.loadtxt(SC_matrices_dir_path_1, delimiter=',')
      
    #   index = avg_pcorr_matrix_ts.argsort(axis=1)
    #   n_rois = avg_pcorr_matrix_ts.shape[0]
    
    #   for i in range(n_rois):
    #     for j in range(n_rois - self.neighbors):
    #       avg_pcorr_matrix_ts[i, index[i, j]] = 0
    #     for j in range(n_rois - self.neighbors, n_rois):
    #       avg_pcorr_matrix_ts[i, index[i, j]] = 1  

      SC_matrix_nx = from_numpy_matrix(avg_pcorr_matrix_ts)         
      pcorr_matrix_data = from_networkx(SC_matrix_nx)     
      filename_without_extension = os.path.splitext(filename)[0]
        
      with open(labels_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
            
        for row in csv_reader:
          if filename_without_extension == row[filename_column]:
            ii=row[data_column]
            print(f"Filename: {filename}, Data: {ii}") 

      lower_triangle = np.tril(ts)
      combine_features = local_degree_profiles(lower_triangle, 2)
      pcorr_matrix_data.x = torch.tensor(combine_features).float()   
      pcorr_matrix_data.y = torch.tensor(int(ii)) 
      pcorr_matrix_data.filename = filename_without_extension
      graphs.append(pcorr_matrix_data)

    data, slices = self.collate(graphs)
    torch.save((data, slices), self.processed_paths[0])


class SZ_FC_K_5_test_plus(InMemoryDataset):
#   def __init__(self, root, transform=None, pre_transform=None, neighbors=5):
  def __init__(self, root, transform=None, pre_transform=None):
    # self.neighbors = neighbors
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_file_names(self):
    return ['data.pt']

  def process(self):
    graphs = []
    filename_column = 0
    data_column = 1

    FC_matrices_dir_path_1= f'{saved_exp_path}/k_5_FC/All/average.csv'

    for i, filename in enumerate(sorted_matching_filenames):

      corr_matrices_dir_path = os.path.join(corr_matrices_dir, filename)
      SC_matrices_dir_path = os.path.join(SC_matrices_dir, filename)
      print(f"Processing files: {corr_matrices_dir_path} and {SC_matrices_dir_path}")
      print(f"Processing files: {corr_matrices_dir_path}")
      
      ts = np.loadtxt(corr_matrices_dir_path, delimiter=',')
      avg_pcorr_matrix_ts = np.loadtxt(FC_matrices_dir_path_1, delimiter=',')
      
    #   index = avg_pcorr_matrix_ts.argsort(axis=1)
    #   n_rois = avg_pcorr_matrix_ts.shape[0]
    
    #   for i in range(n_rois):
    #     for j in range(n_rois - self.neighbors):
    #       avg_pcorr_matrix_ts[i, index[i, j]] = 0
    #     for j in range(n_rois - self.neighbors, n_rois):
    #       avg_pcorr_matrix_ts[i, index[i, j]] = 1  

      SC_matrix_nx = from_numpy_matrix(avg_pcorr_matrix_ts)         
      pcorr_matrix_data = from_networkx(SC_matrix_nx)     
      filename_without_extension = os.path.splitext(filename)[0]
        
      with open(labels_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
            
        for row in csv_reader:
          if filename_without_extension == row[filename_column]:
            ii=row[data_column]
            print(f"Filename: {filename}, Data: {ii}") 

      lower_triangle = np.tril(ts)
      combine_features = local_degree_profiles(lower_triangle, 2)
      pcorr_matrix_data.x = torch.tensor(combine_features).float()   
      pcorr_matrix_data.y = torch.tensor(int(ii)) 
      pcorr_matrix_data.filename = filename_without_extension
      graphs.append(pcorr_matrix_data)

    data, slices = self.collate(graphs)
    torch.save((data, slices), self.processed_paths[0])

