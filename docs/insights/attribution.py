import sys
import numpy as np
import torch
import torchvision
from climex.models.model_cbam_resnet import CBAMResCNNEncoder
from captum.attr import IntegratedGradients, NoiseTunnel
from climex.models.utils import check_int_gr_0
import warnings
from typing import Union
from enum import Enum


def get_attribution(model, inputs, sign,
                    outlier_perc, target=-1, noise_tunnel=False, nt_samples=5, nt_type='smoothgrad_sq'):
    """
    Calculate the attributions with Integrated Gradients (captum)

    Args:
        model (torchvision.models object):  pytorch model to address task of weather pattern detection
        inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
        sign (string, optional): Chosen sign of attributions to visualize. Supported
                    options are:
                    1. `positive` - Displays only positive pixel attributions.
                    2. `absolute_value` - Displays absolute value of
                       attributions.
                    3. `negative` - Displays only negative pixel attributions.
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:
                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples
                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.
                        For outputs with > 2 dimensions, targets can be either:
                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.
                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.
                        Default: -1
        noise_tunnel (bool): If true, noise tunnel is applied to Integrated Gradient.
        nt_samples (int, optional):  The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `nt_samples` is not provided.
        nt_type (string, optional): Smoothing type of the attributions.
                        `smoothgrad`, `smoothgrad_sq` or `vargrad`
                        Default: `smoothgrad` if `type` is not provided.

    Returns:
        attributions with Integrated Gradients in the shape of the map [16,39]
        (if more than one picture: average of the pictures)

    """
    # arguments check
    check_args_get_attribution(model, inputs, sign, outlier_perc, target, noise_tunnel, nt_samples, nt_type)

    # create integrated gradient model
    integrated_gradients = IntegratedGradients(model)

    # add noise if noise_tunnel= True
    if noise_tunnel:
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig = noise_tunnel.attribute(inputs, internal_batch_size=1,
                                                 nt_samples=nt_samples, nt_type=nt_type, target=target)
    else:
        # calculate attributions of channels
        attributions_ig = integrated_gradients.attribute(inputs, target=target, internal_batch_size=1, n_steps=50)

    # average of all inputs = global attribution
    global_avg_attribution = torch.mean(attributions_ig, 0, True)

    # transpose to format for visualisation
    attr = np.transpose(global_avg_attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # normalize attributions for visualisation
    normalized_attributions = normalize_image_attr(attr, sign, outlier_perc)

    return normalized_attributions


def check_args_get_attribution(model, inputs, sign, outlier_perc, target, noise_tunnel, nt_samples, nt_type):
    """
    Check arguments of 'get_attribution'
    """

    # checks model
    if not isinstance(model, (torchvision.models.resnet.ResNet, CBAMResCNNEncoder)):
        sys.exit("`inputs` must be a torchvision.models.")

    # checks inputs
    if not isinstance(inputs, (tuple, torch.Tensor)):
        sys.exit("`inputs` must be tuple or tensor.")

    # checks sign
    if sign not in ['positive', 'absolute_value', 'negative']:
        sys.exit("`sign` must be 'positive', 'absolute_value' or 'negative'.")

    # checks outlier_perc
    if outlier_perc <= 0:
        sys.exit("`sample_size` must be greater than 0.")

    if outlier_perc >= 100:
        sys.exit("`sample_size` must be smaller than 100.")

    # checks target
    num_samples = inputs.shape[0]
    if isinstance(target, list) or (
            isinstance(target, torch.Tensor) and torch.numel(target) > 1
    ):
        assert num_samples == len(target), (
            "The number of samples provied in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )

    # checks noise_tunnel
    if type(True) != type(noise_tunnel):
        sys.exit("`noise_tunnel` must be a boolean.")

    # checks nt_samples
    check_int_gr_0('nt_samples', nt_samples)

    # checks nt_type
    if nt_type not in ['smoothgrad', 'smoothgrad_sq', 'vargrad']:
        sys.exit("`nt_type` must be 'smoothgrad', 'smoothgrad_sq' or 'vargrad'.")

    # Methods for normalizing attribution----------------------------------------------

class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _normalize_scale(attr: np, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values: np, percentile: Union[int, float]):
    # given values should be non-negative
    assert 0 <= percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def normalize_image_attr(
        attr: np, sign: str, outlier_perc: Union[int, float] = 2):
    attr_combined = np.sum(attr, axis=2)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)