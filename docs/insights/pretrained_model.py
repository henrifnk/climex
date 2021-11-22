import sys
import torchvision
import torch
from climex.models.train_cbam_resnet import train_cbam_resnet
from climex.models.train_resnet import train_resnet
from climex.models.utils import check_common_args, check_int_gr_0
from climex.models.model_cbam_resnet import CBAMResCNNEncoder, BasicBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_type='ResNet',
              pretrained=True,
              model_dir='\\docs\\insights\\models\\tests\\ResNet',
              batch_size=256,
              epochs=100, patience=7, lr=0.01, kernel_size=7, stride=2, padding=3, use_weight=True,
              weights=None, n_years_val=10, n_years_test=10, splitting_method='sequential', season=None,
              path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc'):
    """

    The function get_model returns either a the trained model (callable function) based on the model dir path
    (pretrained = True)  or trains a new model with the given parameters.

    Args:
        model_type (char): It can be set to 'ResNet' or 'CBAMResNet': which model should be loaded or trained.
        pretrained (bool): True, if there is a trained model in model_dir and it does not need to be trained again.
        model_dir: Directory in which the trained model and the log file are saved.
        batch_size (int): Defines the number of samples that will be propagated through the network in one epoch.
        epochs (int): Number of epochs.
        patience (int): How many epochs to wait with early stopping after validation loss increases.
        lr (float): Learning rate.
        kernel_size (int): The size of the kernel.
        stride (int): The size of stride.
        padding (int): The size of padding.
        use_weight (bool): True for weighted loss function.
        weights (list of 3 floats or None): Only needs to be set if use_weight=True.
            List with 3 floats containing the manual weights for the loss function.
            If None then weights are computed depending on class frequency.
        n_years_val (int): The number of years to include in the validation set.
        n_years_test (int): The number of years to include in the test set.
        splitting_method (char): The splitting method executed. Must be 'sequential' or 'random'.
        season (char or None): Setting it to None means that all images will be loaded.
                                 Alternatively, it can be set to 'winter' or 'summer' to only load images
                                 from winter or summer months respectively.
        path_to_data: Path to the nc file with the entire dataset.

    Returns: model with trained parameters (callable function which creates outputs out of inputs)

    """
    # Arguments checks
    check_args_get_model(model_type, pretrained, batch_size, epochs, patience, lr, kernel_size, stride,
                         padding, use_weight, weights, n_years_val, n_years_test, splitting_method,
                         season, path_to_data, model_dir)

    # if model not already pretrained, first train model
    if not pretrained:
        train_model(model_type,
                    model_dir,
                    batch_size, epochs, patience, lr, kernel_size, stride, padding, use_weight, weights,
                    n_years_val, n_years_test, splitting_method, season, path_to_data)

    # retrun model and with trained paramters
    return get_trained_model(model_type, model_dir=model_dir,
                             kernel_size=kernel_size, stride=stride, padding=padding)


def get_trained_model(model_type, model_dir, kernel_size, stride, padding):
    """

     The function get_trained_model () returns the callable model function given the model_type and the model_dir.

    Args:
        model_type (char): It can be set to 'ResNet' or 'CBAMResNet': which model should be loaded or trained.
        model_dir: Directory in which the trained model and the log file are saved.
        kernel_size (int): The size of the kernel.
        stride (int): The size of stride.
        padding (int): The size of padding.

    Returns: model with trained parameters (callable function which creates outputs out of inputs)

    """
    # load the model architectures of the models
    if model_type == 'ResNet':
        # Use ResNet18
        model = torchvision.models.resnet18()
        # Use 2-dim kernel, since our image only has 2 channel
        model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        num_ftrs = model.fc.in_features
        # Set the size of each output sample to 3
        model.fc = torch.nn.Linear(num_ftrs, 3)

    elif model_type == 'CBAMResNet':
        model = CBAMResCNNEncoder(BasicBlock).to(device)

    else:
        sys.exit("`model_type` must be 'ResNet' or 'CBAMResNet'.")

    # load trained parameters into the architecture
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir + 'ResNet.pt', map_location=torch.device('cpu')))
    # switch model to evaluation mode
    model.eval()

    return model


def train_model(model_type, model_dir, batch_size, epochs, patience, lr, kernel_size, stride,
                padding, use_weight, weights, n_years_val, n_years_test, splitting_method, season, path_to_data):
    """
    The function train_model calls the train function depending on the model_type (train & save the model in model_dir)

    Args:
        model_type (char): It can be set to 'ResNet' or 'CBAMResNet': which model should be loaded or trained.
        model_dir: Directory in which the trained model and the log file are saved.
        batch_size (int): Defines the number of samples that will be propagated through the network in one epoch.
        epochs (int): Number of epochs.
        patience (int): How many epochs to wait with early stopping after validation loss increases.
        lr (float): Learning rate.
        kernel_size (int): The size of the kernel.
        stride (int): The size of stride.
        padding (int): The size of padding.
        use_weight (bool): True for weighted loss function.
        weights (list of 3 floats or None): Only needs to be set if use_weight=True.
            List with 3 floats containing the manual weights for the loss function.
            If None then weights are computed depending on class frequency.
        n_years_val (int): The number of years to include in the validation set.
        n_years_test (int): The number of years to include in the test set.
        splitting_method (char): The splitting method executed. Must be 'sequential' or 'random'.
        season (char or None): Setting it to None means that all images will be loaded.
                                 Alternatively, it can be set to 'winter' or 'summer' to only load images
                                 from winter or summer months respectively.
        path_to_data: Path to the nc file with the entire dataset.
    """

    # check arguments
    check_args_train_model(model_type, model_dir, batch_size, epochs, patience, lr, kernel_size, stride,
                           padding, use_weight, weights, n_years_val, n_years_test, splitting_method, season,
                           path_to_data)

    if model_type == 'ResNet':
        # call train fuction of ResNet
        train_resnet(batch_size=batch_size, epochs=epochs, patience=patience, lr=lr, kernel_size=kernel_size,
                     stride=stride, padding=padding, use_weight=use_weight,
                     weights=weights, n_years_val=n_years_val, n_years_test=n_years_test,
                     splitting_method=splitting_method, season=season,
                     path_to_data=path_to_data,
                     model_dir=model_dir)

    elif model_type == 'CBAMResNet':
        # call train fuction of CBAMResNet
        train_cbam_resnet(batch_size=batch_size, epochs=epochs, patience=patience, lr=lr, use_weight=use_weight,
                          weights=weights, n_years_val=n_years_val, n_years_test=n_years_test,
                          splitting_method='sequential', season=season,
                          path_to_data=path_to_data,
                          model_dir=model_dir)

    else:
        sys.exit("`model_type` must be 'ResNet' or 'CBAMResNet'.")


def check_args_get_model(model_type, pretrained, batch_size, epochs, patience, lr, kernel_size, stride,
                         padding, use_weight, weights, n_years_val, n_years_test, splitting_method,
                         season, path_to_data, model_dir):

    """Argument checks for `get_model` function.
    """

    # Checks for common args (that appear in all models)
    check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                      splitting_method, season, path_to_data, model_dir)

    # Checks for model_type
    if model_type not in ['ResNet', 'CBAMResNet']:
        sys.exit("`model_type` must be 'ResNet' or 'CBAMResNet'.")

    # Checks for pretrained
    if not isinstance(pretrained, bool):
        sys.exit("`pretrained` must be a bool.")

    # Checks for kernel_size
    check_int_gr_0('kernel_size', kernel_size)

    # Checks for stride
    check_int_gr_0('stride', stride)

    # Checks for padding
    check_int_gr_0('padding', padding)


def check_args_train_model(model_type, model_dir, batch_size, epochs, patience, lr, kernel_size, stride,
                           padding, use_weight, weights, n_years_val, n_years_test, splitting_method, season,
                           path_to_data):

    """Argument checks for `train_model` function.
    """

    # Checks for common args (that appear in all models)
    check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                      splitting_method, season, path_to_data, model_dir)

    # Checks for model_type
    if model_type not in ['ResNet', 'CBAMResNet']:
        sys.exit("`model_type` must be 'ResNet' or 'CBAMResNet'.")

    # Checks for kernel_size
    check_int_gr_0('kernel_size', kernel_size)

    # Checks for stride
    check_int_gr_0('stride', stride)

    # Checks for padding
    check_int_gr_0('padding', padding)