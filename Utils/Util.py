from time import perf_counter
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from tensorflow.keras.utils import to_categorical
import pandas as pd
import json
import dgl


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation_network(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        if dataset_str == 'nell':
            with open("datasets/ind.{}.0.001.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        else:
            with open("datasets/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    if dataset_str == 'nell':
        test_idx_reorder = parse_index_file("datasets/ind.{}.0.001.test.index".format(dataset_str))
    else:
        test_idx_reorder = parse_index_file("datasets/ind.{}.test.index".format(dataset_str))

    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_knowledge_graph(dataset_str="nell", label_rate=0.001, normalization="AugNormAdj", cuda=True):
    """
    Load Knowledge Graph Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datasets/ind.{}.{}.{}".format(dataset_str.lower(), label_rate, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/ind.{}.{}.test.index".format(dataset_str, label_rate))

    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(allx.shape[0], len(graph))
    isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
    tx_extended[test_idx_range - allx.shape[0], :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
    ty_extended[test_idx_range - allx.shape[0], :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


# Generate the antecedent parameters by k-means
def get_antecedent_parameters(features_train, rules):
    kmeans = KMeans(n_clusters=rules, random_state=0)
    # Tensor in GPU to CPU
    features_train = features_train.cpu()
    features_train = kmeans.fit(features_train)
    virtual_centers_nodes = features_train.cluster_centers_
    return virtual_centers_nodes


# Firing Level -Norm
def mu_norm(mus):
    mus_norm = []
    mus_total = 0
    for mu in mus:
        mus_total = mu + mus_total
    for mu in mus:
        mus_norm.append(mu / mus_total)
    return mus_norm


def Mu_Norm_List(features, centers, idx_train, idx_val, idx_test):
    mu_norm_list_train = []
    for feature in features[idx_train]:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list_train.append(mu)

    mu_norm_list_train = torch.tensor(mu_norm_list_train)
    mu_norm_list_train = mu_norm_list_train
    mu_norm_list_train = mu_norm_list_train.unsqueeze(2)
    mu_norm_list_val = []
    for feature in features[idx_val]:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list_val.append(mu)
    mu_norm_list_val = torch.tensor(mu_norm_list_val)
    mu_norm_list_val = mu_norm_list_val
    mu_norm_list_val = mu_norm_list_val.unsqueeze(2)
    mu_norm_list_test = []
    for feature in features[idx_test]:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list_test.append(mu)
    mu_norm_list_test = torch.tensor(mu_norm_list_test)
    mu_norm_list_test = mu_norm_list_test
    mu_norm_list_test = mu_norm_list_test.unsqueeze(2)

    return mu_norm_list_train, mu_norm_list_val, mu_norm_list_test


def Mu_Norm_List_Cluster(features, centers):
    mu_norm_list = []
    for feature in features:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list.append(mu)
    mu_norm_list = torch.tensor(mu_norm_list)
    mu_norm_list = mu_norm_list.cuda()
    mu_norm_list = mu_norm_list.unsqueeze(2)
    return mu_norm_list


def GFS_node_Preprocess(features, adj, rules, alpha, beta, gamma):
    orig_fea = features
    grah_raw_res_fea = orig_fea
    out_fea = alpha * grah_raw_res_fea
    t = perf_counter()
    for i in range(rules):
        features = torch.spmm(adj, features)
    out_fea = features
    X_ce = orig_fea
    X_se = orig_fea
    X_ee = orig_fea
    out_fea = out_fea + gamma * (X_ce + X_se + X_ee)
    precompute_time = perf_counter() - t
    return out_fea, precompute_time


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
        data['test_index']


def load_reddit_data(normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("./datasets/")
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index


def load_clustering_citation_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_data(dataset, repeat, self_loop):
    path = 'datasets/data_geom/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
    train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
    val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))

    label_oneHot = torch.FloatTensor(to_categorical(l))

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj + self_loop * sp.eye(sadj.shape[0])
    nsadj = torch.FloatTensor(sadj.todense())

    return nsadj, features, label, label_oneHot, idx_train, idx_val, idx_test


# load node regression
def read_files(src):
    edges_df = pd.read_csv(src['edges_csv_file'])
    target_df = pd.read_csv(src['target_csv_file'])
    with open(src['features_json_file'], 'r') as json_file:
        node_features = json.load(json_file)
    return edges_df, node_features, target_df


dataset_files = {
    'dataset_files_chameleon': {
        'target_csv_file': './datasets/wikipedia/chameleon/musae_chameleon_target.csv',
        'edges_csv_file': './datasets/wikipedia/chameleon/musae_chameleon_edges.csv',
        'features_json_file': './datasets/wikipedia/chameleon/musae_chameleon_features.json'
    },
    'dataset_files_squirrel': {
        'target_csv_file': './datasets/wikipedia/squirrel/musae_squirrel_target.csv',
        'edges_csv_file': './datasets/wikipedia/squirrel/musae_squirrel_edges.csv',
        'features_json_file': './datasets/wikipedia/squirrel/musae_squirrel_features.json'
    },
    'dataset_files_crocodile': {
        'target_csv_file': './datasets/wikipedia/crocodile/musae_crocodile_target.csv',
        'edges_csv_file': './datasets/wikipedia/crocodile/musae_crocodile_edges.csv',
        'features_json_file': './datasets/wikipedia/crocodile/musae_crocodile_features.json'
    }
}


def find_inlier_indexes(target_df):
    # print('\ntarget values distribution:\n',target_df['target'].describe())

    # Calculate Q1 and Q3
    Q1 = target_df['target'].quantile(0.25)
    Q3 = target_df['target'].quantile(0.75)
    # print('Qs:', Q1, Q3)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # print('bounds:', lower_bound, upper_bound)

    # Filter and remove outliers
    df_filtered = target_df[(target_df['target'] >= lower_bound) & (target_df['target'] <= upper_bound)]
    indexes_to_keep = df_filtered['id'].tolist()

    # Outliers
    # print(target_df[(target_df['target'] < lower_bound) | (target_df['target'] > upper_bound)])

    return indexes_to_keep


def filter_target_df(target_df, indexes_to_keep):
    filtered_target_df = target_df.copy()

    # Filter rows where 'id' is in the list
    filtered_target_df = filtered_target_df[filtered_target_df['id'].isin(indexes_to_keep)]

    # Create a mapping of old indexes to new indexes
    new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(indexes_to_keep)}

    # Update 'id' column with new indexes
    filtered_target_df['id'] = filtered_target_df['id'].map(new_index_mapping)

    # Reset the index of the DataFrame
    filtered_target_df.reset_index(drop=True, inplace=True)

    return filtered_target_df, new_index_mapping


def filter_edges_df(edges_df, indexes_to_keep, new_index_mapping):
    # Create a boolean mask for filtering based on both 'id1' and 'id2'
    mask = edges_df['id1'].isin(indexes_to_keep) & edges_df['id2'].isin(indexes_to_keep)

    # Filter rows based on the mask
    filtered_edges_df = edges_df[mask].copy()  # Create a copy of the filtered DataFrame

    # Update 'id1' and 'id2' columns with new indexes
    filtered_edges_df['id1'] = filtered_edges_df['id1'].map(new_index_mapping)
    filtered_edges_df['id2'] = filtered_edges_df['id2'].map(new_index_mapping)

    # Reset the index of the DataFrame
    filtered_edges_df.reset_index(drop=True, inplace=True)

    return filtered_edges_df


def filter_node_features(node_features_dict, new_index_mapping, indexes_to_keep):
    # Filter and update the node_features dictionary without loops
    filtered_node_features = {str(new_index_mapping[int(old_index)]): features for old_index, features in
                              node_features_dict.items() if
                              int(old_index) in indexes_to_keep}

    return filtered_node_features


def remove_outliers(target_df, edges_df, node_features_dict):
    # print('\nOutliers to Remove:')
    indexes_to_keep = find_inlier_indexes(target_df)

    target_df_filtered, new_index_mapping = filter_target_df(target_df, indexes_to_keep)
    edges_df_filtered = filter_edges_df(edges_df, indexes_to_keep, new_index_mapping)
    node_features_dict_filtered = filter_node_features(node_features_dict, new_index_mapping, indexes_to_keep)
    return target_df_filtered, edges_df_filtered, node_features_dict_filtered


def normalize_target_values(target_df):
    # print(min(target_values), max(target_values))
    df_min_max_scaled = target_df.copy()
    min_val, max_val = df_min_max_scaled['target'].min(), df_min_max_scaled['target'].max()
    df_min_max_scaled['target'] = (df_min_max_scaled['target'] - min_val) / (max_val - min_val)
    # print(df_min_max_scaled['target'])
    return df_min_max_scaled


def create_binary_features_tensor(node_feats):
    # print('\nAdd features and targets to the graph...')

    # Determine the number of unique integer features in the dataset
    unique_features = set(feature for features_list in node_feats.values() for feature in features_list)
    num_unique_features = len(unique_features)
    # print('number of unique features:', len(unique_features))

    # Create an empty tensor to hold the binary node features
    num_nodes = len(node_feats)
    binary_features_tensor = torch.zeros(num_nodes, num_unique_features, dtype=torch.float32)

    # Iterate through the node indices and features in the node_features dictionary
    for node_idx, features_list in node_feats.items():
        # Iterate through the integer features and set the corresponding cell to 1
        for feature in features_list:
            feature_index = list(unique_features).index(feature)
            binary_features_tensor[int(node_idx)][feature_index] = 1

    # print('Done.')
    return binary_features_tensor


def create_masks(num_nodes, t_ratio):
    # Define the ratio of nodes for training (it can be adjusted as needed)
    train_ratio = t_ratio  # 60% for training, 20% for validation, 20% for testing

    # Set the random seed for reproducibility (optional)
    np.random.seed(0)
    # Generate random indices for shuffling the nodes
    rand_indices = np.random.permutation(num_nodes)

    # Split the nodes based on the defined ratios
    train_idx = rand_indices[:int(train_ratio * num_nodes)]
    val_idx = rand_indices[int(train_ratio * num_nodes):int((train_ratio + 0.2) * num_nodes)]
    test_idx = rand_indices[int((train_ratio + 0.2) * num_nodes):]
    # print(train_idx)
    # print(val_idx)
    # print(test_idx)

    # Create train, validation, and test masks
    # train_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))
    # val_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))
    # test_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))
    #
    # train_mask[train_idx] = True
    # val_mask[val_idx] = True
    # test_mask[test_idx] = True

    return train_idx, val_idx, test_idx


def create_graph(edges_df, binary_node_features, target_df):
    # Extract source and destination nodes from edges
    src_nodes = edges_df['id1'].tolist()
    dst_nodes = edges_df['id2'].tolist()

    # Create a DGL graph
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(binary_node_features))
    graph = dgl.add_self_loop(graph)  # Optionally add self-loops if needed

    # Save a plot of graph
    # plot_graph(graph)

    # Add the binary feature vectors to the DGL graph's node data
    graph.ndata['feature'] = binary_node_features
    # print([graph.ndata['feature'][2098][2111]])

    # Add target values to the graph
    target_values = target_df['target'].tolist()
    graph.ndata['target'] = torch.tensor(target_values, dtype=torch.float32)  # dtype could be int64?

    # Boxplot the target values and their distribution
    # boxplot_targets(target_values)

    # Create and Add the masks to the DGL graph's node data
    num_nodes = graph.number_of_nodes()
    # print(num_nodes)

    train_idx, val_idx, test_idx = create_masks(num_nodes, t_ratio=0.6)
    # print(len(train_idx))
    # graph.ndata['train_idx'], graph.ndata['val_idx'], graph.ndata['test_idx'] = train_idx, val_idx, test_idx
    # print(train_idx)

    return graph, train_idx, val_idx, test_idx


def get_graph_details(graph):
    features = graph.ndata['feature']
    targets = graph.ndata['target']
    graph_details = [features, targets]
    return graph_details


def show_graph_details(graph, graph_details):
    features, targets, train_idx, val_idx, test_idx = graph_details
    print('\nGraph:', graph)
    print('\nfeatures:')
    print(features)
    print('\ntargets:')
    print(targets)
    print('\nTrain, Validation, and Test Masks:')
    print(train_idx)
    print(val_idx)
    print(test_idx)


def load_data_regression(dataset_name):
    # Load wiki dataset
    edges_df, node_features, target_df = read_files(dataset_files['dataset_files_' + dataset_name])

    # Preprocess & Remove outliers
    target_df, edges_df, node_features = remove_outliers(target_df, edges_df, node_features)

    # Normalize target values

    target_df = normalize_target_values(target_df)

    # Convert dictionary node features to tensor binary vectors
    binary_node_features = create_binary_features_tensor(node_features)

    # print(binary_node_features)
    # Create graph from data files
    graph, train_idx, val_idx, test_idx = create_graph(edges_df, binary_node_features, target_df)

    # Define and show graph parameters
    graph_details = get_graph_details(graph)
    # show_graph_details(graph, graph_details)
    features, targets = graph_details
    return features, targets, train_idx, val_idx, test_idx


def RMSE_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))
