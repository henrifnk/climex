import random
import sys
import torch
import torchvision
from torch.autograd import Variable
from climex.data.load_data import load_data
from climex.models.model_cbam_resnet import CBAMResCNNEncoder

from climex.models.utils import check_int_gr_0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input_data(batch_size=1, season=None,
                   path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc',
                   sample_size=0.1, seed=123):
    """
    This function loads the data with load_data and extracts a certain percentage of the batches.

    Args:
        batch_size (int): Defines the number of samples that it was propagated through the network in one epoch.
        season (char or None): Setting it to None means that all images will be loaded.
                                 Alternatively, it can be set to 'winter' or 'summer' to only load images
                                 from winter or summer months respectively.
        path_to_data: Path to the nc file with the entire dataset.
        sample_size: deciding number of samples from original dataset (in percentage). Default: 10%
        seed: seed for sampling the data

    Returns: sample from the input data in the shape [2,16,39, number of inputs]

    """
    # check arguments
    check_args_get_input_data(batch_size, season, path_to_data, sample_size, seed)

    # Load, split and preprocess data (images)
    # as we do not need a split here: just one year for testing & validation
    train_loader, test_loader, val_loader = load_data(
        batch_size=batch_size,
        n_years_val=1,
        n_years_test=1,
        splitting_method='random',
        season=season,
        path_to_data=path_to_data
    )

    # get number of batches
    number_of_batches = len(train_loader)

    # set seed for sampling & reproducibility
    random.seed(a=seed, version=2)

    # sample  certain percentage (default: 10% of the data) of batches
    sample_batches = random.sample(range(0, number_of_batches), round(number_of_batches * sample_size))

    # initialize inputs and labels
    inputs = torch.empty(0)
    labels = torch.empty(0)

    # add all sampled batches into input & label
    for batch in sample_batches:
        input_batch = list(train_loader)[batch][0]
        label_batch = list(train_loader)[batch][1]

        inputs = torch.cat((inputs, input_batch), 0)
        labels = torch.cat((labels, label_batch), 0)

    # transform to format for input to models
    inputs, labels = Variable(inputs).to(device, dtype=torch.float), Variable(labels).to(device, dtype=torch.int64)

    return inputs, labels


def get_input_label(inputs, model, pred_label):
    """
    Extracts inputs which are predicted to be a specific label.
    Args:
        inputs (torch.tensor): output of the first part of the get_input_data
        model (torchvision.models object):  pytorch model to address task of weather pattern detection
        pred_label (torch.Tensor): filter for the inputs: which input creates this label with the model
    Returns: input where label is equal to given label
    """
    inputs = inputs.to(device)

    # initialize input_label for all inputs with predicted label
    input_label = torch.empty(0)
    input_label = input_label.to(device)

    prediction_indices = torch.empty(0)
    prediction_indices = prediction_indices.to(device)

    # number of images of the variable inputs
    length = inputs.shape[0]

    # for all inputs add to input_label if predicted label is equal pred_label
    for i in range(0, length):

        # index of input
        index = torch.tensor(i)
        index = index.to(device)

        # get input at index i
        one_input = torch.index_select(inputs, dim=0, index=index)
        one_input = one_input.to(device)

        # predict output for input at index i
        outputs = model(one_input)
        _, prediction_index = torch.max(outputs.data, 1)

        # if predicted index is equal to wanted label (pred_label) add to input_label
        if prediction_index == pred_label:
            input_label = torch.cat((input_label, one_input), 0)
            prediction_indices = torch.cat((prediction_indices, prediction_index), 0)

    # which indices have this predicted label
    return input_label, prediction_indices


def check_args_get_input_data(batch_size, season, path_to_data, sample_size, seed):
    """Argument checks for `get_input_data` function.
    """
    # Checks for batch_size
    check_int_gr_0('batch_size', batch_size)

    # Checks for season
    if season not in [None, 'summer', 'winter']:
        sys.exit("`season` must be None, 'winter' or 'summer'.")

    if not path_to_data[-3:] == ".nc":
        sys.exit("Can only load netcdf files with file extension '.nc'.")

    if not isinstance(sample_size, (int, float)):
        sys.exit("`sample_size` must be an integer.")

    if sample_size <= 0:
        sys.exit("`sample_size` must be greater than 0.")

    if sample_size > 1:
        sys.exit("`sample_size` must be smaller than 1.")

    check_int_gr_0('seed', seed)


def check_args_get_input_label(inputs, model, pred_label):
    """Argument checks for `get_input_label` function.
    """

    # checks inputs
    if not isinstance(inputs, (tuple, torch.tensor)):
        sys.exit("`inputs` must be tuple or tensor.")

    # checks model
    if not isinstance(model, (torchvision.models.resnet.ResNet, CBAMResCNNEncoder)):
        sys.exit("`inputs` must be a torchvision.models.")

    # checks pred_label
    if not isinstance(pred_label, torch.Tensor):
        sys.exit("`pred_label` must be tuple or tensor.")