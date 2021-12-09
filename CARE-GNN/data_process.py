from utils import sparse_to_adjlist
from scipy.io import loadmat
import sys
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz, load_npz, coo_matrix

"""
	Read data and save the adjacency matrices to adjacency lists
"""


if __name__ == "__main__":

	# btc_homo = load_npz("btc_train/train_adj_mat.npz")
	# btc_r1 = load_npz("btc_train/leq_25_relation_adjmat.npz")
	# btc_r2 = load_npz("btc_train/leq_40_ge_25_relation_adjmat.npz")
	# btc_r3 = load_npz("btc_train/geq_40_relation_adjmat.npz")

	# prefix = 'btc_train/'

	# sparse_to_adjlist(btc_r1, prefix + 'btc_leq_25_adjlists.pickle')
	# sparse_to_adjlist(btc_r2, prefix + 'btc_leq_40_ge_25_adjlists.pickle')
	# sparse_to_adjlist(btc_r3, prefix + 'btc_geq_40_adjlists.pickle')
	# sparse_to_adjlist(btc_homo, prefix + 'btc_homo_adjlists.pickle')
	# print('done')


	prefix = 'data/'

	yelp = loadmat('data/YelpChi.mat')
	net_rur = yelp['net_rur']
	net_rtr = yelp['net_rtr']
	net_rsr = yelp['net_rsr']
	yelp_homo = yelp['homo']
	print(yelp['features'].todense().A.shape)
	# sys.exit()


	sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

	amz = loadmat('data/Amazon.mat')
	net_upu = amz['net_upu']
	net_usu = amz['net_usu']
	net_uvu = amz['net_uvu']
	amz_homo = amz['homo']

	sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')
