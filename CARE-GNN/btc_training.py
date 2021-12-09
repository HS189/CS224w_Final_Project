import time
import os
import random
import argparse
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='yelp', help='The dataset name. [yelp, amazon]')
parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=100, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=50, help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')

# other args
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

import torch
torch.manual_seed(3)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'run on {args.data}')

np.random.seed(args.seed)
random.seed(args.seed)

def load_btc_data():
	labels = np.load('btc_train/train_classes.npy')
	feat_data = np.load('btc_train/train_features.npy')
	prefix = 'btc_train/'
	with open(prefix + 'btc_homo_adjlists.pickle', 'rb') as file:
		homo = pickle.load(file)
	with open(prefix + 'btc_leq_25_adjlists.pickle', 'rb') as file:
		relation1 = pickle.load(file)
	with open(prefix + 'btc_leq_40_ge_25_adjlists.pickle', 'rb') as file:
		relation2 = pickle.load(file)
	with open(prefix + 'btc_geq_40_adjlists.pickle', 'rb') as file:
		relation3 = pickle.load(file)

	return [homo, relation1, relation2, relation3], feat_data, labels

# load graph, feature, and label
[homo, relation1, relation2, relation3], feat_data, labels = load_btc_data()
# labels[labels == 1] = 0
labels[labels == 2] = 0
print(labels)

# print(np.count_nonzero(labels == 1))
# print(np.count_nonzero(labels == 0))
# import sys
# sys.exit()

ts_col = feat_data[:,0]
idx_train=np.where(ts_col <= 34)[0]
# idx_val = np.where((ts_col > 25) & (ts_col <= 34))[0]


idx_test=np.where(ts_col > 34)[0]
idx_train = list(idx_train)
# idx_val = list(idx_val)
idx_test = list(idx_test)

y_train = labels[idx_train]
y_test = labels[idx_test]
# y_val = labels[idx_val]

y_train = list(y_train)
# y_val = list(y_val)
y_test = list(y_test)

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
	features.cuda()

# set input graph
if args.model == 'SAGE':
	adj_lists = homo
else:
	adj_lists = [relation1, relation2, relation3]

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# build one-layer models
if args.model == 'CARE':
	intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
					  step_size=args.step_size, cuda=args.cuda)
elif args.model == 'SAGE':
	agg1 = MeanAggregator(features, cuda=args.cuda)
	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

if args.model == 'CARE':
	gnn_model = OneLayerCARE(2, inter1, args.lambda_1)
elif args.model == 'SAGE':
	# the vanilla GraphSAGE model as baseline
	enc1.num_samples = 5
	gnn_model = GraphSage(2, enc1)

if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
times = []
performance_log = []

# train the model
for epoch in range(args.num_epochs):
	# randomly under-sampling negative nodes for each epoch
	sampled_idx_train = undersample(train_pos, train_neg, scale=2)
	rd.shuffle(sampled_idx_train)

	# send number of batches to model to let the RLModule know the training progress
	num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
	if args.model == 'CARE':
		inter1.batch_num = num_batches

	loss = 0.0
	epoch_time = 0

	# mini-batch training
	for batch in range(num_batches):
		start_time = time.time()
		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
		batch_nodes = sampled_idx_train[i_start:i_end]
		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		epoch_time += end_time - start_time
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

	# testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:
		if args.model == 'SAGE':
			test_sage(idx_test, y_test, gnn_model, args.batch_size)
			print('above is wrong, this is correct:')
			modified_test_sage(idx_test, y_test, gnn_model)
		else:
			# gnn_auc, label_auc, gnn_recall, label_recall = test_care(idx_test, y_test, gnn_model, args.batch_size)
			best_threshold = best_val_thresh_care(idx_train, y_train, gnn_model)
			# best_threshold = 0.56
			print('best F1 thresh:', best_threshold)
			modified_test_care(idx_test, y_test, gnn_model, thresh=best_threshold)
			# performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])
