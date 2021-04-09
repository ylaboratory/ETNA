# proj-template
simple template for ylab projects

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
