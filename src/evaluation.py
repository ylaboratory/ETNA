# this file includes functions to evaluate network alignment result
import heapq
import random
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn import metrics


def evaluate_all(result_matrix, ontology_matrix, weight_matrix):
    '''
    take two set of aligned genes and gene-term dictonary for two organism
    return a list of jaccard index for the aligned genes
    '''
    result_flatten = result_matrix.flatten()
    ontology_flatten = ontology_matrix.flatten()
    weight_matrix = weight_matrix.flatten()
    fpr, tpr, _ = metrics.roc_curve(ontology_flatten,
                                    result_flatten,
                                    sample_weight=weight_matrix)
    return metrics.auc(fpr, tpr), metrics.average_precision_score(ontology_flatten,
                                                                  result_flatten,
                                                                  average='weighted',
                                                                  sample_weight=weight_matrix)


def go_term_jaccard_weight(align_x, align_y,
                           g1_node2index, g2_node2index,
                           org1_dict, org2_dict, weight_matrix):
    '''
    take top aligned genes from two organisms, two node2index dict,
    two gene2term dict, and weight matrix. Calculate the weighted
    jaccard index score for those pairs.
    '''
    ret = []
    for i in range(len(align_x)):
        terms1 = org1_dict[align_x[i]]
        terms2 = org2_dict[align_y[i]]
        inters = terms1.intersection(terms2)
        union = terms1.union(terms2)
        weight = weight_matrix[g1_node2index[align_x[i]]][
            g2_node2index[align_y[i]]]
        if union:
            score = len(inters) / len(union) * weight
        else:
            score = 0
        ret.append(score)
    return ret


def go_term_jaccard_weigted_info(align_x, align_y,
                                 org1_dict, org2_dict,
                                 info_dict):
    '''
    take top aligned genes from two organisms, two node2index dict,
    two gene2term dict, and information dict for go term pairs.
    Calculate the jaccard index score weighted by go term info.
    '''
    ret = []
    for i in range(len(align_x)):
        terms1 = org1_dict[align_x[i]]
        terms2 = org2_dict[align_y[i]]
        inters = terms1.intersection(terms2)
        union = terms1.union(terms2)
        inters = [info_dict[x] for x in inters]
        union = [info_dict[y] for y in union]
        if union:
            s = np.sum(inters) / np.sum(union)
        else:
            s = 0
        ret.append(s)
    return ret


def top_align(result_matrix, org1_index2node, org2_index2node, n):
    '''
    take result matrix, two index2node dict, and a given number n
    return top n number of aligned genes
    '''
    result_flatten = result_matrix.flatten()
    order = heapq.nlargest(n, range(len(result_flatten)), result_flatten.take)
    order = np.array(order)
    row = (order / result_matrix.shape[1]).astype(int)
    column = (order % result_matrix.shape[1]).astype(int)
    row = [org1_index2node[x] for x in row]
    column = [org2_index2node[y] for y in column]
    return row, column


def get_resnik_score(terms1, terms2, score_dict):
    '''
    take two set of GO terms and resnik score dictionary
    return the max resnik score between them
    '''
    s = [0]
    for t1 in terms1:
        for t2 in terms2:
            if t1 < t2:
                key = t1 + '_' + t2
                s.append(score_dict[key])
    return max(s)


def edge_alignment(network1, network2, links):
    '''
    calculate the number of edges in the first network
    that are aligned to edges in the second network.
    '''
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


def node_alignment(network1, network2, links):
    '''
    calculate the number of edges in the second network
    that both nodes have corresponding nodes in the first network
    '''
    node_mapping = defaultdict(set)
    for a, b in links:
        node_mapping[b].add(a)
    edges2 = network2.edges()
    edges2 = [(x, y) for x, y in edges2
              if len(node_mapping[x]) > 0 and len(node_mapping[y]) > 0]
    return set(edges2)


def compare_with_random(term, s_matrix, g1_term2indexes, g2_term2indexes,
                        cutoff=1000, ite=100):
    '''
    t-test between scores for genes within a go term and scores from
    randomly sampled genes with same length
    '''
    g1_true_index = list(g1_term2indexes[term])
    g2_true_index = list(g2_term2indexes[term])
    if len(g1_true_index) > cutoff or len(g2_true_index) > cutoff:
        return -1
    true_matrix = s_matrix[g1_true_index][:, g2_true_index]
    totally_random = []
    for _ in range(ite):
        g1_random_index = random.sample(list(range(s_matrix.shape[0])),
                                        len(g1_true_index))
        g2_random_index = random.sample(list(range(s_matrix.shape[1])),
                                        len(g2_true_index))
        totally_random.append(s_matrix[g1_random_index][
                              :, g2_random_index].flatten())
    return stats.ttest_ind(true_matrix.flatten(), np.mean(totally_random, axis=0))[1]


def filter_top_pairs(xs, ys, ortholog):
    '''
    filter out ortholog pairs from top alignment
    '''
    top_pair = [(xs[i], ys[i]) for i in range(len(xs))
                if (xs[i], ys[i]) not in ortholog]
    return [x for x, y in top_pair], [y for x, y in top_pair]


def filter_score(xs, ys, ortholog, score):
    '''
    assign -1 to ortholog pairs in score list
    '''
    indexes = [i for i in range(len(score)) if (xs[i], ys[i]) in ortholog]
    score = np.array(score)
    score[indexes] = -1
    return score
