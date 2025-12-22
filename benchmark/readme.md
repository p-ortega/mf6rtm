# Benchmarks

These Jupyter Notebooks run classic, well-know reference models using this MF6RTM package to compare model outputs with those produced by PHT3D and PHREEQC (for example 4):

1. **Example 1: Engesgaard and Kipp 1992. 1D Precipitation and Dissolution Fronts.** A one-dimensional model domain in which an aqueous water composition that is in equilibrium with two minerals, calcite and dolomite, is successively replaced, i.e., flushed by water of a different chemical composition, leading to multiple precipitation-dissolution fronts. 
2. **Example 2: Walter 1994. 1D migration of AMD precipitation & dissolution fronts.** A one-dimensional, purely inorganic redox problem that  demonstrates the evolution of some important geochemical processes that occur when acidic mine drainage (AMD) leaches into an anaerobic carbonate aquifer.
3. **Example 2: Walter 1994. 2D migration of AMD precipitation & dissolution fronts.** A two-dimensional version of Example 2.
4. **Example 4: Parkhurst and Appelo 2013 (PHREEQC-3 Example 11) 1D Cation Exchange.** Cation exchange column flushing of a sodium-potassium nitrate solution with calcium chloride.
5. **Example 5: Appelo 1998. Pyrite Oxidation**  Modelling of an oxidation experiment with marine pyrite-containing sediments.
6. **Example 6: Parkhurst and Appelo 2013 (PHREEQC-3 Example 11) Cation Exchange.** 3D variant of Cation Exchange with DISV grid

## Install Development Environment

### Original Approach

This approach installs dependencies similarly as they would be from an install from PyPi.

Follow the recommendations for [Developing](README.md#developing) on the main README.md for this repo.


### Alternate Approach using only Conda

This approach relies entirely on Conda to pull all external dependencies and to install this repo in develop mode. 

This approach also uses an alternate environment file: `benchmark/environment.yml`.

Follow these steps to install using the [conda](https://docs.conda.io/en/latest/) package manager.

#### 1. Install Miniconda or Anaconda Distribution

We recommend installing the light-weight [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) that includes Python, the [conda](https://conda.io/docs/) environment and package management system, and their dependencies.

NOTE: Follow conda defaults to install in your local user director. DO NOT install for all users, to avoid substantial headaches with permissions.

If you have already installed the [**Anaconda Distribution**](https://www.anaconda.com/download), you can use it to complete the next steps, but you may need to [update to the latest version](https://docs.anaconda.com/free/anaconda/install/update-version/).

If you are on Windows, we recommend initializing conda for all your command prompt terminals, by opening the "Anaconda Prompt" console and typing this command:

```shell
conda init --all
```

#### 2. Clone or Download this Repository

From this Github page, click on the green "Code" dropdown button near the upper right. Select to either "Open in GitHub Desktop" (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of this repo in any convenient location on your computer.

#### 3. Create a Conda Environment for this Repository

We recommend creating a custom virtual environment with the same software dependencies that we've used in development and testing, as listed in the [`environment.yml`](environment.yml) file. 

Create a project-specific environment using this [conda](https://conda.io/docs/) command in your terminal or Anaconda Prompt console. If necessary, replace `environment.yml` with the full file pathway to the `environment.yml` file in the local cloned repository.

```shell
conda env create --file environment.yml
```

Alternatively, use the faster [`libmamba` solver](https://conda.github.io/conda-libmamba-solver/getting-started/) with:

```shell
conda env create -f environment.yml --solver=libmamba
```

Activate the environment using the instructions printed by conda after the environment is created successfully.

To update your environment run the following command:  

```shell
conda env update --file environment.yml --solver=libmamba --prune 
```


#### 4. Install this Package in Develop Mode

To access this `mf6rtm` package in this conda environment, it is necessary to have a path to your clone in this environment's `sites-packages` directory (i.e. something like `$HOME/path/to/miniconda3/envs/<env_name>/lib/python3.11/site-packages/conda.pth`).

The easiest way to do this is to use the [`conda develop`](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) command in the console or terminal like this, replacing '/path/to/module/src' with the full file pathway to the local cloned Clearwater-riverine repository:

```shell
conda develop '/path/to/module/src'
```

You should now be able to run the examples and create your own Jupyter Notebooks!