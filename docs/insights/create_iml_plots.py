import os
import sys

import torch

from climex.models.utils import check_int_gr_0
from docs.insights.input import get_input_data, get_input_label
from docs.insights.plot_iml_regions import check_args_path_to_data, plot_iml_regions
from docs.insights.pretrained_model import get_model


def create_iml_plots(path_to_data,
                     model_dir='docs/insights/tests/models/ResNet/',
                     model_type='ResNet',
                     pretrained=True,
                     sample_size=0.1,
                     fig_path='docs/insights/plots/',
                     labels=(torch.tensor(1), torch.tensor(2), None),
                     fig_names=('iml-resnet-tm-nt', 'iml-resnet-trm-nt', 'iml-resnet-all-nt'),
                     seed=123,
                     noise_tunnel=True):
    """
    Creates dor different predicted labels for a specific model the plots
    Args:
        path_to_data: Path to the nc file with the entire dataset.
        model_dir: Directory in which the trained model and the log file are saved.
        model_type (char): It can be set to 'ResNet' or 'CBAMResNet': which model should be loaded or trained.
        pretrained (bool): True, if there is a trained model in model_dir and it does not need to be trained again.
        sample_size: deciding number of samples from original dataset (in percentage). Default: 10%
        fig_path(str): Path to the figure.
        labels (tuple of torch.tensor or torch.tensor): If None, the plot shows the average attribution wrt all inputs
            if a specific label: only attributions for inputs where the model creates this labels as prediction
            in this case:
            * create plot for pred_label: torch.tensor(1) predicted image "Tiefmitteleuropa" / 11 / TM
            * create plot for pred_label: torch.tensor(2) predicted image "Trog Mitteleuropa" / 17 / TRM
        fig_names (tuple of str or str): name of the saved figures corresponding to the variable 'labels'.
            'fig_name' and 'labels must have the same length.
        seed: seed for sampling the data
        noise_tunnel (bool): If true, noise tunnel is applied to Integrated Gradient.

    Returns: Saved plots

    """

    # get trained ResNet Model
    model = get_model(model_type=model_type, pretrained=pretrained,
                      model_dir=model_dir)

    # get sampled input data
    input_data, input_label = get_input_data(path_to_data=path_to_data,
                                             sample_size=sample_size, seed=seed)

    for index, label in enumerate(labels):
        if label is not None:
            input_of_label, target_of_label = get_input_label(input_data, model, pred_label=label)
        else:
            input_of_label, target_of_label = input_data, input_label
        plot_iml_regions(input_of_label, model, noise_tunnel=noise_tunnel, target=target_of_label,
                         fig_path=fig_path, fig_name=fig_names[index])


def check_args_create_iml_plots(path_to_data, model_dir, model_type, pretrained, sample_size,
                                fig_path, labels, fig_names, seed, noise_tunnel):
    """
    Argument checks for `create_iml_plots´
    """

    check_args_path_to_data(path_to_data)

    # Checks for model_dir
    if not isinstance(model_dir, str):
        sys.exit("`model_dir` must be a string.")

    # Checks for model_type
    if model_type not in ['ResNet', 'CBAMResNet']:
        sys.exit("`model_type` must be 'ResNet' or 'CBAMResNet'.")

    # Checks for pretrained
    if not isinstance(pretrained, bool):
        sys.exit("`pretrained` must be a bool.")

    if not isinstance(sample_size, (int, float)):
        sys.exit("`sample_size` must be an integer.")

    if sample_size <= 0:
        sys.exit("`sample_size` must be greater than 0.")

    if sample_size > 1:
        sys.exit("`sample_size` must be smaller than 1.")
    # checks fig_path
    if not isinstance(fig_path, str):
        sys.exit("`fig_path` must be a string.")

    if not os.path.exists(fig_path):
        sys.exit("`fig_path` does not exists.")

    # checks for labels
    if not isinstance(labels, (tuple, torch.tensor)):
        sys.exit("`labels` must be tuple or torch.tensor.")

    # checks for fig_names
    if not isinstance(fig_names, (tuple, str)):
        sys.exit("`fig_names` must be tuple or str.")

    if not len(fig_names) == len(labels):
        sys.exit("`fig_names` must be the same length as ´labels´")

    check_int_gr_0('seed', seed)

    # checks noise_tunnel
    if not isinstance(noise_tunnel, bool):
        sys.exit("`noise_tunnel` must be a boolean.")