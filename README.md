# ETNA: Embeddings to Network Alignment

This repository contains the scripts to run the ETNA method and corresponding analysis
described in the Li et al. paper,
_Joint embedding of biological networks for cross-species functional alignment_.

## Citation

> Joint embedding of biological networks for cross-species functional alignment.
> Li L, Dannenfelser R, Zhu Y, Hejduk N, Segarra S, Yao V. Bioinformatics. August 2023.
> [https://doi.org/10.1093/bioinformatics/btad529](https://doi.org/10.1093/bioinformatics/btad529)

## About

Model organisms are widely used to better understand the molecular
causes of human disease. While sequence similarity greatly aids this transfer,
sequence similarity does not imply functional similarity, and thus, several
current approaches incorporate protein-protein interactions to help map
findings between species. Existing transfer methods either formulate the
alignment problem as a matching problem which pits network features
against known orthology, or more recently, as a joint embedding problem.
Here, we propose a novel state-of-the-art joint embedding
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

ETNA's method is roughly divided into 3 main parts:

  1. training an autoencoder to embed PPI networks
  2. aligning the embeddings between species via ortholog anchors
  3. scoring gene pairs across organisms with cosine similarity in the embedding

These steps are implemented in `src/algorithms/ETNA.py`. The demo
jupyter notebook (`/src/demo.ipynb`) illustrates running ETNA to align
two PPI networks from _S. cerevisiae_ and _S. pombe_.

## Usage

This project uses conda to manage the required packages and setup a
virtual environment. Once conda is installed on your machine get started
by setting up the virtual environment.

```sh
conda env create -f env.yml
conda activate etna
```

We have created demo code for running and evaluating ETNA on two PPI networks.
To run the demo start up a jupyter notebook with the following command:

```sh
jupyter lab --port=8888
```

Navigate in a browser to running notebook at `http://localhost:8888`
and open the `src` folder to load and run `demo.py`.
