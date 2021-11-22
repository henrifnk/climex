import torch
import os
import numpy as np
import sys

def add_gnoise(video, mean = 0, var = 0.3):
    """add gaussian noise to pictures
    """
    img, channel, lats, lons = video.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (img, channel, lats, lons))
    return video + gauss

def adjust_learning_rate(base_lr, optimizer, epoch):
    """Adjusts the learning rate in an torch optimizer object twice,
     originating from base learning rate by epoch

     Args:
        base_lr: (float) initial learning rate
        optimizer: (torch.optim.Optimizer)torch optimizer object
        epoch: (int) current model epoch
    """
    if epoch < int(epoch * 0.6):
        lr = base_lr
    elif int(epoch * 0.6) <= epoch < int(epoch * 0.75):
        lr = base_lr * 0.1
    else:
        lr = base_lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class EarlyStopping:
    """Saves patterns to recognize early stopping the training

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
            Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        name (string):  model name
            Default: 'ResNet.pt'
    """

    def __init__(self, model_dir, patience=7, verbose=False, delta=0, name='ResNet.pt'):
        """Initialize early stopping class
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_dir = model_dir
        self.name = name

    def __call__(self, val_loss, model, model2=None):
        """Call early stopping.
        Args:
            val_loss (float):                   Validation loss, calculated in evaluation on validation data
            model (torchvision.models object):  pytorch model to adress task of weather pattern detection
            model2 (torchvision.models object): pytorch model to adress task of weather pattern detection
                                                only used if model consists of an encoder and a decoder model
                                                Default: None
        """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model2)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model2)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model2):
        """Saves model when validation loss decrease.
        """
        # Create a directory to the model if non exists
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.val_loss_min = val_loss
        # ResNet: Only one model needs to be stored
        if model2 is None:
            torch.save(model.state_dict(), self.model_dir + self.name)

        # LSTM: Store two models; Encoder and Decoder
        else:
            torch.save(model.state_dict(), self.model_dir + "CNN.pt")
            torch.save(model2.state_dict(), self.model_dir + "RNN.pt")


def setup_weights(train_loader, weights):
    """ Set Up and configuration for manual and automatically setting weights before model training
    Args:
        train_loader (Data Loader object):  Pytorch training dataset
        weights (list of 3 floats or None): List with 3 floats containing the manual weigths for the loss function.
                                            If None then weigths are computed depending on class frequency.

    Returns:
        Torch.FloatTensor of weights
    """
    # Manual weighted loss
    if weights is not None:
        weights = [float(item) for item in weights]
    # Computed weighted loss
    else:
        weights = train_loader.dataset.get_label_weight()

    # Convert weigths to pytorch format
    if torch.cuda.is_available():
        weights = torch.FloatTensor(weights).cuda()
    else:
        weights = torch.FloatTensor(weights).cpu()
    return weights


def check_int_gr_0(var, val):
    """ Checks if val is an integer greater than 0 and throws error if not.

    Arguments:
        var (char): Variable name.
        val: Variable value
    """
    if (not isinstance(val, int)):
        sys.exit("`" + var + "` must be an integer.")
    if (val <= 0):
        sys.exit("`" + var + "` must be greater than 0.")


def check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                      splitting_method, season, path_to_data, model_dir):
    """ Implements checks for common arguments (that appear in all models).

    Arguments
            See e.g. `?train_resnet`.
    """
    # Checks for batch_size
    check_int_gr_0('batch_size', batch_size)

    # Checks for epochs
    check_int_gr_0('epochs', epochs)

    # Checks for patience
    check_int_gr_0('patience', patience)

    # Checks for lr
    if (not isinstance(lr, float)):
        sys.exit("`lr` must be a float.")
    if (lr <= 0):
        sys.exit("`lr` must be greater than 0.")

    # Checks for use_weight
    if (not isinstance(use_weight, bool)):
        sys.exit("`use_weight` must be a bool.")

    # Checks for weights
    weights_is_list_len3 = isinstance(weights, list) and len(weights) == 3
    weights_is_float_list_len3 = weights_is_list_len3 and all(isinstance(x, float) for x in weights)
    if not (weights_is_float_list_len3 or weights is None):
        sys.exit("`weights` must be a float list of len 3 or None.")

    # Checks for n_years_val and n_years_test
    check_int_gr_0('n_years_val', n_years_val)
    check_int_gr_0('n_years_test', n_years_test)

    # Checks for splitting_method
    if (splitting_method not in ['sequential', 'random']):
        sys.exit("`splitting_method` must be 'sequential' or 'random'.")

    # Checks for season
    if (season not in [None, 'summer', 'winter']):
        sys.exit("`season` must be None, 'winter' or 'summer'.")

    # Checks for path_to_data
    if (not isinstance(path_to_data, str)):
        sys.exit("`path_to_data` must be a string.")

    if (not path_to_data[-3:] == ".nc"):
        sys.exit("Can only load netcdf files with file extension '.nc'.")

    if (not os.path.exists(path_to_data)):
        sys.exit("`path_to_data` does not exists.")

    # Checks for model_dir
    if (not isinstance(model_dir, str)):
        sys.exit("`model_dir` must be a string.")
