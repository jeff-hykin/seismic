# SEISMIC: _State Estimation/Integration of Sensors from Miniature Insect Connectome_

This repository is meant to include code supporting the paper "Continous State Estimation from Synapse Constrained Connectivity", presented at WCCI2022, and available online at [IEEE Xplore]() and [arXiv]().

In short, it enables the optimization of weights of a recurrent neural network, where the structure is derived from a circuit in _D. melanogaster_ that performs heading direction estimation and has properties of a ring attractor network. This circuit is responsible for orientation estimation/integration within the fruit fly, and thus we optimized to integrate an angular velocity over time. 

This repository uses synapse-level connectome data from [hemibrain](https://www.janelia.org/project-team/flyem/hemibrain) to generate a network model. As part of this network generation, we found a new pattern of connections that was not included in previous models. This code provides a way to evaluate the functional properties of  a synapse-level generated network model for continuous state estimation, including the role of these new connection patterns.


# Installation

This package is mainly based on [pytorch](https://pytorch.org/). The installation process should be quite simple, but might be slightly more involved depending on your hardware.

It is recommneded to use a virtual enviroment, such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/).

To install use pip:

```
pip install git+https://github.com/aplbrain/seismic.git
```

This should provide a minimal install of the library, without any dependencies for plotting or legacy code.

The following extra depencies can be installed to enable additional functionality
* `plotting`: Dependencies for plotting
* `neuprint` : Dependencies to pull new connectomic data from [Neuprint](https://neuprint.janelia.org/)
* `all` : All extra dependencies

You can provide an extra by modifying the install command, for example:
```
pip install git+https://github.com/aplbrain/seismic.git#egg=neuroaiengines[examples]
```

# Downloading connectivity data (optional)

**NOTE: This section requires an install using `['all']` or `['neuprint']` (see installation instructions)**

The connectivity matrices are stored on [neuprint](https://neuprint.janelia.org/), Janelia Research Campus' repository for hemibrain. Pre-downloaded data is included in this repository for ease (see `neuroaiengines/networks/*.csv`).

To pull the raw synapse-level connecitivity data from neuprint, you first need to acquire an API token on the [Neuprint website](https://neuprint.janelia.org/). Next, run the following script with your token:

```
python neuroaiengines/networks/pull_neuprint_data.py <token>
```

This pulls the data directly from neuprint and creates a file `hemibrain_conn_df_both.csv`, which includes connections from both `PEN(a)` and `PEN(b)`. If you wish to download additional data with just `PEN(a)` or `PEN(b)`, you can run 
```
python neuroaiengines/networks/pull_neuprint_data.py <token> --pen-a --pen-b
```
This enables additional analyses.
# Examples
Examples can be found in `./examples/<experiment_name>`. Two examples are included, which are used in the paper. `examples/synapse_level` is the example that uses the synapse level network including novel connections. `examples/null_offset` is a network that only includes previously identified connections.

Running the examples involves three steps:

## 1. Generating training/testing data

You can generate data using `./examples/<experiment_name>/generate_data.py`. This creates three files with self explanatory names under `./examples/<experiment_name>/data/`. 

The metadata files are pickled dicts that give ground truth orientation data etc.

## 2. Training a model

To train a model, run one of the training scripts: `./examples/<experiment_name>/train.py`

This creates a file `./examples/<experiment_name>/data/model.pt`, which contains the pytorch model state-dict. It also saves training metadata under `./runs` in tensorboard format. Training information can be visualized with tensorboard.

```
tensorboard --logdir=./runs
```


## 3. Evaluating a model

To train a model, run one of the testing scripts: `./examples/<experiment_name>/test.py`

This creates a file `testing_outputs.pkl` that contains the testing outputs, similar to the training output file.





# Generate data/figures for the paper

See the [figure generation notebook](./notebooks/make_paper_figures.ipynb) to see how the figures in the paper were generated. The figures have been slightly changed from the figures in the paper to avoid copyright issues.

# Code structure
The code is structued as follows

* `neuroaiengines`

    This python package is the main package used for training the ring attractors.
    * `networks`

        This folder includes code for generating the weight matrices from EM connectivity.
        * `ring_attractor.py`

            Pytorch model of the ring attractor.
        * `__init__.py`
        
            Code to generate weight matrices from connectivity pulled from Neuprint.

        * `offset_utils.py`

            Methods that pull data from Neuprint, as well as some plotting utilities
        
        * `pull_data.py`

            Script to pull data from Neuprint and save it in a proper format.
        * `hemibrain_conn_df_{a,b,both}.csv`

            Pre-downloaded data files
        
    * `optimization`

        Code related to the optimization.

        * `datasets.py`

            Iterable datasets based on several different methods of generation.
        * `torch.py`

            Pytorch utilities, particularly loss functions, and a class that implements (truncated) backpropgation through time. 
        * `utils.py`

            Miscellaneous utilities for training, such as high level functions.
        * `simulator.py`

            Contains code for generating more complex velocity trajectory examples, for example in a 2D environment. Also includes some code for generating landmark angles from an environment.

    * `utils`

        A bunch of utilties.

        * `angles.py`

            Utils related to converting angles, normalization, etc.
        * `plotting.py`

            Utilities for plotting.
        * `signals.py`

            Utilities for working with/generating EPG bumps, preferred neuron directions etc.
        * `transforms.py`

            Utilities for modifying weight matrices.
* `notebooks`

    This directory includes Jupyter notebooks to render the figures used in the paper, as well as a simple use of the ring attractor outside of a training/testing script.
* `examples`

    Includes training/testing scripts for the two example ring attractor structures, see the above section for details on running.


