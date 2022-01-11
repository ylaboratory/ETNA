# ETNA: Embedding to Network Alignment

This repo contains the scripts to run the ETNA method and corresponding analysis described in the Li et al. paper, _Joint embedding of biological networks for cross-species functional alignment_


## Citation
> Discriminatory power of combinatorial antigen recognition in cancer T cell therapies.
Dannenfelser R, Allen G, VanderSluis B, Koegel AK, Levinson S, Stark SR, Yao V, Tadych A, Troyanskaya OG, Lim WA. Cell Systems. 2020. [https://doi.org/10.1016/j.cels.2020.08.002](https://doi.org/10.1016/j.cels.2020.08.002)
<!-- (DOI badge for later?[![DOI](https://zenodo.org/badge/126377943.svg)](https://zenodo.org/badge/latestdoi/126377943)) -->

This repo includes a basic `.gitignore` with common files to exclude, but this should obviously be pared down / additional files should be added as necessary.

There is also support for [super-linter](https://github.com/github/super-linter) as a [GitHub action](https://docs.github.com/en/free-pro-team@latest/actions), which essentially just means that all code will be automatically linted on push / when PRs are opened. Make sure all checks pass!

The directory structure is inspired by [this article](https://medium.com/outlier-bio-blog/a-quick-guide-to-organizing-data-science-projects-updated-for-2016-4cbb1e6dac71), which is based off of this [classic article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424) on organizing projects, and makes a good starting point for projects.

## conda environment
The `env.yml` file should be updated accordingly for projects that use python, so that a new conda environment can be easily installed using the following command:
```sh
conda env create -f env.yml
```

Per usual, to activate the environment:
```sh
conda activate new_env_name
```

If the environment is already set up, to update it for new dependencies / resources:
```sh
conda env update -n new_env_name -f env.yml --prune
```

Note that the `--prune` flag will tell conda to remove any dependencies that may no longer be required in the environment.
