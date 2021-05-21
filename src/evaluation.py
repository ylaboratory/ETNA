#this file includes functions to evaluate network alignment result

import numpy as np
from collections import defaultdict
from sklearn import metrics
import heapq


'''
read in gmf file and return a dictionary with
gene keys and set of go terms as value
'''

def read_in_gmf(file_name):
    ret_dict = defaultdict(set)
    with open(file_name, 'r') as f:
        for line in f:
            tokens = line.split('\t')
            tokens = [x.strip() for x in tokens]
            for gene in tokens[2:]:
                ret_dict[gene].add(tokens[0])

    return ret_dict

'''
take result score matrix and label matrix 
return AUROC and AUPRC
'''

def evaluate_all(result_matrix, ontology_matrix):
    result_flatten = result_matrix.flatten()
    ontology_flatten = ontology_matrix.flatten()
    fpr, tpr, thresholds = metrics.roc_curve(ontology_flatten, result_flatten)
    return metrics.auc(fpr, tpr), metrics.average_precision_score(ontology_flatten, result_flatten)

'''
take two set of aligned genes and gene-term dictonary for two organism
return a list of jaccard index for the aligned genes
'''


def go_term_jaccard(align_x, align_y, org1_dict, org2_dict):
    n = len(align_x)
    return [len(org1_dict[align_x[i]].intersection(org2_dict[align_y[i]])) /
            len(org1_dict[align_x[i]].union(org2_dict[align_y[i]]))
            if len(org1_dict[align_x[i]].intersection(org2_dict[align_y[i]])) > 0
            else 0
            for i in range(n)]

'''
take result matrix, two index2node dict, and a given number n
return top n number of aligned genes
'''

def top_align(result_matrix, org1_index2node, org2_index2node, n):
    result_flatten = result_matrix.flatten()
    order = heapq.nlargest(n, range(len(result_flatten)), result_flatten.take)
    order = np.array(order)
    row = (order / result_matrix.shape[1]).astype(int)
    column = (order % result_matrix.shape[1]).astype(int)
    row = [org1_index2node[x] for x in row]
    column = [org2_index2node[y] for y in column]
    return row, column

'''
take two set of GO terms and resnik score dictionary
return the max resnik score between them
'''

def get_resnik_score(terms1, terms2, score_dict):
    s = [0]
    for t1 in terms1:
        for t2 in terms2:
            if t1 < t2:
                key = t1 + '_' + t2
                s.append(score_dict[key])
    return max(s)

'''
calculate the number of edges in the first network
that are aligned to edges in the second network.
'''

def edge_alignment(network1, network2, links):
    node_mapping = defaultdict(set)
    for a, b in links:
        node_mapping[a].add(b)
    edges1 = network1.edges()
    aligned_edges = set()
    for x, y in edges1:
        a = node_mapping[x]
        b = node_mapping[y]
        for x in a:
            for y in b:
                aligned_edges.add((min(x, y), max(x, y)))
    edges2 = network2.edges()
    edges2 = set([(min(x, y), max(x, y)) for x, y in edges2])
    return aligned_edges.intersection(edges2)

'''
calculate the number of edges in the second network
that both nodes have corresponding nodes in the first network
'''

def node_alignment(network1, network2, links):
    node_mapping = defaultdict(set)
    for a, b in links:
        node_mapping[b].add(a)
    edges2 = network2.edges()
    edges2 = [(x, y) for x, y in edges2
              if len(node_mapping[x]) > 0 and len(node_mapping[y]) > 0]
    return set(edges2)
