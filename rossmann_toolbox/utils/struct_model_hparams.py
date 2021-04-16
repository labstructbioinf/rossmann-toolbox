##########################################################
##########################################################
############## DEEPLIGNAD TRAINING PARAMS ################
##########################################################
##########################################################

hparams = dict()
#input number of edge features
hparams['in_dim_e'] = 20
#input number of node features
hparams['in_dim_n'] = 20

#last layer output dim - aka number of classifier classes (NAD, NADP, SAM and FAD)
hparams['n_classes'] = 4

########################################################
################## GRAPH LAYERS CONFIG #################

#activation function used in scaling edge attetion scores
hparams['attention_scaler'] = 'sigmoid'

#number of repeated Edge GAT blocks
hparams['num_blocks'] = 5

#block input node features (dimensionality of the first GAT layer output (block input))
hparams['block_in'] = 64
#block input (and also it's hiden) dimensionality of edge features
hparams['hidden_dim_e'] = 32
#number of heads in each block
hparams['blocks_heads'] = 2
#block output `block_in` must be divisible by `blocks_heads`
hparams['block_out'] =  hparams['block_in'] // hparams['blocks_heads']

# drop rate of edges (not working atm)
hparams['p_attn'] = 0.0
#drop rate of node features (after removing seqvec `p_feat = 0` work best)
hparams['p_feat'] = 0.0

#########################################################
#####################LEARNING PARAMS#####################
#initial learning rate
hparams['learning_rate'] = 1e-3
# Adam optimizer type
hparams['use_amsgrad'] = True
#number of epochs before reducting learning rate (without improvment)
hparams['plateau_patience'] = 8
# L2 regularization term
hparams['reg_term'] = 4e-2

# classes weights - not used at this moment
hparams['class_weights'] = None
# do not change 
hparams['loss_size_avarage'] = True
#value used 
hparams['f1_type'] = 'macro'
