# This file includes auxiliary functions for loading data and evaluation.

import numpy as np

def node2index(nx_g):
    '''
    map from node name to node index
    '''
    return {j:i for i,j in enumerate(nx_g.nodes())}

def index2node(nx_g):
    '''
    map from node index to node name
    '''
    return {i:j for i,j in enumerate(nx_g.nodes())}

def anchor_idx(anchor, g1, g2):
    '''
    map node pairs from node name to node index
    '''
    idx1 = node2index(g1)
    idx2 = node2index(g2)
    anchor1 = [idx1[item[0]] for item in anchor]
    anchor2 = [idx2[item[1]] for item in anchor]
    return anchor1, anchor2


def sparse_anchor(anchor, g1, g2):
    '''
    generate sparse anchors that nodes in graphs only appear in
    at most one anchor
    '''
    org1_anchor_dict = set()
    org2_anchor_dict = set()
    sparse_anchor = []
    for a, b in anchor:
        if a not in org1_anchor_dict and b not in org2_anchor_dict:
            org1_anchor_dict.add(a)
            org2_anchor_dict.add(b)
            sparse_anchor.append([a, b])
            org1_anchor_dict.update(set(g1.neighbors(a)))
            org2_anchor_dict.update(set(g2.neighbors(b)))
    return sparse_anchor

def normalize(matrix, actions):
    '''
    normalize matrix
    '''
    for action in actions:
        if action == 'unit':
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1
            matrix = (matrix.T/norms).T
        elif action == 'center':
            matrix = matrix - np.mean(matrix,0)
    return matrix
