# Deep Learning Template

This is the PyTorch based template for deep learning project. It define some useful deep learning components (file):

- `model` directory contain all element about model, including network, basic-network and metric to evaluate how network performance
- `data` directory contain training and testing data
- `dataloader` contain function that fetch training and validation and testing dataset
- `experiments` contain all model trained weights, hyper-parameter all in `params.json` file
- `utils` contain some useful function, including setting logging, plot learning curves, and save params
- `build-dataset.py` file are used to download dataset and pre-process it
- `config.py` contain some configuration, for example, the name of training loss file
- `evaluate.py` contain code to e√valuate the model
- `run.py` contain code to run the model to precessing the new inputs
- `search-hyperparams.py`\* : search for better hyper-params.
- `synthesize_results.py`\* : format the training process output
- `train.py`: the code used to train model.

User should modify different files according need.

\*: the file is borrowed from [cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples/tree/master?)
