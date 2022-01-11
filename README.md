# ETNA: Embedding to Network Alignment

This repo contains the scripts to run the ETNA method and corresponding analysis
described in the Li et al. paper,
_Joint embedding of biological networks for cross-species functional alignment_.

## Citation

> Joint embedding of biological networks for cross-species functional alignment
Li L, Dannenfelser R, Zhu Y, Hejduk N, Segarra S, Yao V. BioRxiv. 2022.
[add the doi](https://doi.org/10.1016/j.cels.2020.08.002)
<!-- (DOI badge for later?
	[![DOI](https://zenodo.org/badge/126377943.svg)]
	(https://zenodo.org/badge/latestdoi/126377943)) -->

## About

Model organisms are widely used to further the understanding of human molecular
mechanisms and the dysregulations that result in disease. While sequence
similarity greatly aids this transfer, sequence similarity does not imply
functional similarity, and thus, several current approaches incorporate
protein-protein interactions to help map findings between species. Existing
transfer methods either formulate the alignment problem as a matching problem
which pits network features against known orthology, or more recently, as a
joint embedding problem. Here, we propose a novel state-of-the-art joint embedding
solution: Embeddings to Network Alignment (ETNA). More specifically,
ETNA generates individual network embeddings based on network topological
structures and then uses a Natural Language Processing-inspired cross-training
approach to align the two embeddings using sequence orthologs. The final
embedding preserves both within and between species gene functional
relationships, and we demonstrate that it captures both pairwise and group
functional relevance. In addition, ETNA’s embeddings can be used to transfer genetic
interactions across species and identify phenotypic alignments, laying
the groundwork for potential opportunities for drug repurposing
and translational studies.

ETNA's code is roughly divided into 3 main parts:

  1. training an autoencoder to embed PPI networks
  2. aligning the embeddings between species via ortholog anchors
  3. scoring gene pairs across organisms with cosine similarity in the embedding

Additionally, there is code to compare and evaluate ETNA's pairings with MUNK.
NEED TO EXPAND THIS PART

## Usage

This project uses conda to manage the required packages and setup a
virtual environment. Once conda is installed on your machine get started
by setting up the virtual environment

```sh
conda env create -f env.yml
conda activate etna
```

more usage instructions to come
