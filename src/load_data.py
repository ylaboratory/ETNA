# This file contains functions that load the datasets.
from collections import defaultdict
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd


def keep_distinct_node(g):
    '''
    Keep nodes with distinct neighborhood in the graph
    '''
    A = nx.adjacency_matrix(g).toarray()
    a_dict = defaultdict(set)
    node_list = list(g.nodes())
    for i in range(A.shape[0]):
        a_dict[tuple(A[i])].add(node_list[i])
    distinct_nodes = set()
    for s in a_dict.values():
        if len(s) < 2:
            distinct_nodes.add(list(s)[0])
    removed_nodes = set(g.nodes()).difference(distinct_nodes)
    g.remove_nodes_from(removed_nodes)


def load_ppi(species, k_core=None, verbose=True, lcc=True):
    '''
     Read in the ppi network of the input species.
    '''
    if verbose:
        print('load the ppi network of ' + species)

    parent_dir = Path(__file__).resolve().parent.parent
    f_ppi = str(parent_dir) + '/data/physical_interaction/' + \
        species + '_physical_pairs.txt'
    nx_g = nx.read_edgelist(f_ppi, create_using=nx.DiGraph())
    for edge in nx_g.edges():
        # add edge weights (needed in node2vec)
        nx_g[edge[0]][edge[1]]['weight'] = 1
    if verbose:
        print('read as directed: {} nodes, \
         {} edges'.format(len(nx_g), nx_g.size()))
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))  # remove self-loops
    nx_g.remove_nodes_from(list(nx.isolates(nx_g)))  # remove isolated nodes
    if verbose:
        print('remove selfloop edges: {} nodes, {} edges'.format(
            len(nx_g), nx_g.size()))
    nx_g = nx_g.to_undirected()  # convert to the undirected graph
    if verbose:
        print('convert to undirected: {} nodes, {} edges'.format(
            len(nx_g), nx_g.size()))
    if lcc:
        nx_lcc = nx_g.subgraph(
            max(nx.connected_components(nx_g), key=len))  # keep the lcc
        if verbose:
            print('keep the largest cc: {} nodes, {} edges'.format(
                len(nx_lcc), nx_lcc.size()))
    else:
        nx_lcc = nx_g
    
    if k_core is None:
        nx_kcore = nx_lcc
    else:
        nx_kcore = nx.k_core(nx_lcc, k_core)  # return the k-core
        if verbose:
            print('keep the {}-core: {} nodes,\
                {} edges'.format(k_core, len(nx_kcore), nx_kcore.size()))
            
    node_len = 0
    while node_len != len(nx_kcore.nodes()):
        node_len = len(nx_kcore.nodes())
        keep_distinct_node(nx_kcore) 
        nx_kcore.remove_nodes_from(list(nx.isolates(nx_kcore)))
    if verbose:
        print('return the distinct nodes: {} nodes,\
                {} edges'.format(len(nx_kcore), nx_kcore.size()))
            
    return nx_kcore


def load_functional_network(species, k_core=None, verbose=True, weighted=False, lcc=False):
    '''
     Read in the ppi network of the input species.
    '''
    if verbose:
        print('load the functional network of ' + species)

    parent_dir = Path(__file__).resolve().parent.parent

    f = str(parent_dir) + '/data/functional_network/' + \
        species + '_functional.txt'
    edges = pd.read_csv(f, sep='\t', header=None,
                        dtype={0: str, 1: str, 2: float})
    edges = edges.rename(columns={2: "weight"})
    if weighted:
        nx_g = nx.from_pandas_edgelist(edges, 0, 1, ['weight'])
    else:
        nx_g = nx.from_pandas_edgelist(edges, 0, 1)

    if verbose:
        print('read as: {} nodes, {} edges'.format(len(nx_g), nx_g.size()))

    if lcc:
        nx_g = nx_g.subgraph(
            max(nx.connected_components(nx_g), key=len))  # keep the lcc

    if verbose:
        print('keep the largest cc: {} nodes, {} edges'.format(
            len(nx_g), nx_g.size()))
    if k_core is None:
        return nx_g
    else:
        nx_kcore = nx.k_core(nx_g, k_core)  # return the k-core
        if verbose:
            print('return the {}-core: {} nodes,\
                {} edges'.format(k_core, len(nx_kcore), nx_kcore.size()))
        return nx_kcore


def load_anchor(s1, s2):
    '''
     Read in the anchor links between species s1 and species s2.
     Choices of s1 and s2: (cel,hsa) (cel,mmu) (cel,sce) (hsa,mmu) 
     (hsa,sce) (mmu,sce)
    '''
    parent_dir = Path(__file__).resolve().parent.parent
    f_anchor = str(parent_dir) + '/data/ortholog/' + \
        s1 + '_' + s2 + '_orthomcl.txt'
    df_anchor = pd.read_csv(f_anchor, delimiter='\t')
    df_anchor = df_anchor.sort_values('score', ascending=False)
    anchor = []
    for i in range(len(df_anchor)):
        s1_gene = str(int(df_anchor.iloc[i][s1]))
        s2_gene = str(int(df_anchor.iloc[i][s2]))
        score = df_anchor.iloc[i]['score']
        anchor.append([s1_gene, s2_gene, score])
    return anchor


def filter_anchor(anchor, g1_node2index, g2_node2index, top_k=None):
    '''
     Filter out anchor links whose endpoints do not exist in the ppi networks.
     Output anchor links that are associated with top_k highest scores.
    '''
    filtered = []
    for row in anchor:
        if g1_node2index[row[0]] != -1 and g2_node2index[row[1]] != -1:
            filtered.append(row)
    if top_k is None:
        return filtered
    return filtered[:top_k]


def load_gmt(org):
    '''
    Read in go term and return two dictionary: gene2term and term2gene
    '''
    parent_dir = Path(__file__).resolve().parent.parent

    f_go = str(parent_dir) + '/data/gene_ontology/' + \
        org + '_low_BP_propagated.gmt'
    gene2terms = defaultdict(set)
    term2genes = defaultdict(set)
    with open(f_go, 'r') as f:
        for line in f:
            tokens = line.split('\t')
            tokens = [x.strip() for x in tokens]
            for gene in tokens[2:]:
                gene2terms[gene].add(tokens[0])
                term2genes[tokens[0]].add(gene)
    return gene2terms, term2genes


def load_go_pairs(org1, org2, file_name):
    '''
    Read in gene ontology pair data of input species.
    '''
    parent_dir = Path(__file__).resolve().parent.parent
    f = str(parent_dir) + '/data/' + org1 + '_' + org2 + '/' + file_name
    data = np.loadtxt(f, dtype=str, delimiter='\t')
    data = data[:, [0, 1]]
    print(data.shape)
    return data
