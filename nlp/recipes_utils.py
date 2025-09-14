import pickle
import pandas as pd
import os
import ast
import json
import datetime
import nbformat
import torch
from transformers import (AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, )
from torch.utils.data import Dataset, DataLoader


# DEVICE MANAGEMENT
def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device = torch.device(device)
    print("Device set to:", device)

    if device == "mps":
        # MPS memory management - Used to solve RuntimeError: MPS backend out of memory
        # source: https://pnote.eu/notes/pytorch-mac-setup/
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("PYTORCH_MPS_HIGH_WATERMARK_RATIO set to:", os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])

    return device


def clear_memory(device):
    # Helper function to clear unused memory
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# DATA LOADING AND SAVING
def load_data(filepath):
    data = pd.read_csv(filepath).drop(['Unnamed: 0'], axis=1)
    return data


def save_pickle(dataset, filepath):
    # Save the dataset to a file
    with open(filepath, 'wb') as file:
        pickle.dump(dataset, file)


def load_pickle(filepath):
    # Load the dataset from the file
    with open(filepath, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


# recipes = load_data(dataset_path)
# save_pickle(recipes, pickle_path)


def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15):
    # Split the dataset into train, val and split = 70/15/15 (dummy split with no shuffle)
    train_dataset = data[:int(train_size * len(data))]
    val_dataset = data[int(train_size * len(data)): int((train_size + val_size) * len(data))]
    test_dataset = data[int((train_size + val_size) * len(data)):]
    return train_dataset, val_dataset, test_dataset


# -----------------------------------------------

# RECIPE DATASET PROCESSING
DIR_DATA_PATH = '../data/recipes'
FULL_RECIPES_PATH = f'{DIR_DATA_PATH}/full_dataset.csv'
FULL_RECIPES_PICKLE_PATH = f'../data/recipes.pickle'
TRAIN_PICKLE_PATH = f'{DIR_DATA_PATH}/train.pickle'
VAL_PICKLE_PATH = f'{DIR_DATA_PATH}/val.pickle'
TEST_PICKLE_PATH = f'{DIR_DATA_PATH}/test.pickle'


def load_splits(TRAIN_PICKLE_PATH, VAL_PICKLE_PATH, TEST_PICKLE_PATH, keep_pct=None):
    train = load_pickle(TRAIN_PICKLE_PATH)
    val = load_pickle(VAL_PICKLE_PATH)
    test = load_pickle(TEST_PICKLE_PATH)

    if keep_pct:
        train = train.sample(frac=keep_pct, random_state=42)
        val = val.sample(frac=keep_pct, random_state=42)
        test = test.sample(frac=keep_pct, random_state=42)

    return train, val, test


# CHECKPOINTING FOR NOTEBOOKS
def save_checkpoint(notebook_name, dir='', n_inputs=0):
    if not dir:
        dir = "./checkpoints"
    os.makedirs(dir, exist_ok=True)

    # Load current notebook
    with open(f"./{notebook_name}") as f:
        nb = nbformat.read(f, as_version=4)

    # Create a checkpoint filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{os.path.basename(notebook_name).replace('.ipynb', '')}_{n_inputs}_checkpoint_{timestamp}.ipynb"

    # Save the checkpoint
    with open(f"{dir}/{checkpoint_name}", 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Checkpoint saved as {checkpoint_name}")
    return checkpoint_name


# Example usage: manually trigger a checkpoint save
# save_checkpoint('notebook_Name.ipynb')


def load_checkpoint(checkpoint_name):
    """
    Load a checkpointed notebook and display its contents in the current Jupyter environment.
    """
    with open(f"./checkpoints/{checkpoint_name}", 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    return nb


# Example usage: Load a specific checkpoint
# checkpoint_file = 'checkpoint_path.ipynb'
# loaded_notebook = load_checkpoint(checkpoint_file)


# CHECKPOINTING FOR MODELS

def save_model_checkpoint(model, optimizer, epoch, loss, hyperparams, file_path):
    """
    Save a model checkpoint including the training state and hyperparameters.

    Parameters:
    - model: the model whose parameters to save.
    - optimizer: the optimizer with the current state.
    - epoch: the current epoch number.
    - loss: the loss at the time of saving.
    - hyperparams: dictionary containing all relevant hyperparameters.
    - file_path: the path to save the checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyperparams': hyperparams
    }
    torch.save(checkpoint, f"{file_path}.pt")
    print(f"Checkpoint saved at {file_path}.pt")


def load_model_checkpoint(file_path, model):
    """
    Load a model checkpoint.

    Returns:
    - epoch: the epoch number at which the checkpoint was saved.
    - loss: the loss at the time of saving.
    - hyperparams: the hyperparameters used for training.
    """

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(f"{file_path}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    hyperparams = checkpoint['hyperparams']
    return model, optimizer, epoch, loss, hyperparams


# def download_latest_checkpoint(checkpoint_dir, zip_only=True):
#     latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
#     latest_checkpoint_name = os.path.split(latest_checkpoint_path)[-1]
#     latest_checkpoint_zip_name = latest_checkpoint_name + '.zip'
#
#     print('latest_checkpoint_path: ', latest_checkpoint_path)
#     print('latest_checkpoint_name: ', latest_checkpoint_name)
#     print('---\n')
#
#     print('Checkpoint files:')
#     with zipfile.ZipFile(latest_checkpoint_zip_name, mode='w') as zip_obj:
#         for folder_name, subfolders, filenames in os.walk(checkpoint_dir):
#             for filename in filenames:
#                 if filename.startswith(latest_checkpoint_name):
#                     print('  - ' + filename)
#                     file_path = os.path.join(folder_name, filename)
#                     zip_obj.write(file_path, os.path.basename(file_path))
#     print('---\n')
#     print('Zipped to: ', latest_checkpoint_zip_name)
#
#     if not zip_only:
#         files.download(latest_checkpoint_zip_name)
#
#


# Model setup

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
