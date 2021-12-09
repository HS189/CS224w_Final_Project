import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import os
from tqdm import tqdm

classes_csv = 'elliptic_txs_classes.csv'
edgelist_csv = 'elliptic_txs_edgelist.csv'
features_csv = 'elliptic_txs_features.csv'
data_dir = 'elliptic_bitcoin_dataset'
classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col = 'txId1') # directed edges between transactions
features = pd.read_csv(os.path.join(data_dir, features_csv), header = None, index_col = 0) # features of the transactions


num_features = features.shape[1]
num_tx = features.shape[0]
total_tx = list(classes.index)

# select only the transactions which are labelled
labelled_classes = classes[classes['class'] != 'unknown']
labelled_tx = list(labelled_classes.index)

# to calculate a list of adjacency matrices for the different timesteps

adj_mats = []
features_labelled_ts = []
classes_ts = []

## Single training graph (not split by ts)

features_ts = features[features[1] <= 100]
# features_ts = features[:]
tx_ts = list(features_ts.index)

labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]

# adjacency matrix for all the transactions
# we will only fill in the transactions of this timestep which have labels and can be used for training
adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index = total_tx, columns = total_tx)

edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
for i in range(edgelist_labelled_ts.shape[0]):
    adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1

adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
A = csc_matrix(adj_mat_ts.values)
save_npz("train_adj_mat.npz".format(ts+1), A)
print('saved sparse train adj mat')

features_l_ts = features.loc[labelled_tx_ts]
np.save('train_features.npy', features_l_ts.values) # save
print('saved features for train')
#     new_num_arr = np.load('data.npy') # load

classes_cur = classes.loc[labelled_tx_ts]
np.save('train_classes.npy', classes_cur.values.astype(int).flatten())
print('saved classes for train')