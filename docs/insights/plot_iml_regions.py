import os
import matplotlib.pyplot as plt
import netCDF4
import cartopy
import cartopy.crs as ccrs
import torch
import torchvision
import sys
from matplotlib import colors as c
from docs.insights.attribution import get_attribution
from climex.models.model_cbam_resnet import CBAMResCNNEncoder


def plot_iml_regions(inputs,
                     model,
                     sign='positive',
                     outlier_perc=2,
                     target=-1,
                     noise_tunnel=False,
                     fig_name='iml_region_plot',
                     fig_path='/docs/insights/plots/',
                     path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc'):
    """
    Visualizes attribution for a given image by normalizing attribution values
    of the desired sign (positive, negative, absolute value, or all) and displaying
    them using the desired mode in a matplotlib figure.

    Args:
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
        inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
        model (torchvision.models object):  pytorch model to address task of weather pattern detection
        sign (string, optional): Chosen sign of attributions to visualize. Supported
                    options are:
                    1. `positive` - Displays only positive pixel attributions.
                    2. `absolute_value` - Displays absolute value of
                       attributions.
                    3. `negative` - Displays only negative pixel attributions.
                    Default: `positive`
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
        fig_name(str): Name of the saved figure.(without image type)
        fig_path(str): Path to the figure.
        path_to_data: Path to the nc file with the entire dataset.
    """
    check_args_plot_iml_regions(inputs, model, sign, outlier_perc, target,
                                noise_tunnel, fig_name, fig_path, path_to_data)

    # return nothing, if inputs are empty
    if inputs.shape[0] == 0:
        return

    # get attributions of locations with Integrated Gradients
    attribution = get_attribution(model, inputs, sign, outlier_perc, target, noise_tunnel)

    # get Longitude and Latitude of data
    dataset = netCDF4.Dataset(path_to_data)
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]

    # create plot
    plt.figure()
    plt.figure(figsize=(14, 14))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    # make a color map of fixed colors
    cmap = c.ListedColormap({'#00004c', '#000080', '#0000b3', '#0000e6', '#0026ff', '#004cff', '#0073ff', '#0099ff',
                             '#00c0ff', '#00d900', '#33f3ff', '#73ffff', '#c0ffff', (0, 0, 0, 0), '#ffff00', '#ffe600',
                             '#ffcc00', '#ffb300', '#ff9900', '#ff8000', '#ff6600', '#ff4c00', '#ff2600', '#e60000',
                             '#b30000', '#800000', '#4c0000'})

    # set bounds for colors
    bounds = [-200, -100, -75, -50, -30, -25, -20, -15, -13, -11, -9, -7, -5, -3, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30,
              50, 75, 100, 200]

    c.BoundaryNorm(bounds, ncolors=cmap.N)

    # add attributions to plot (black is highest attribution)
    plt.pcolormesh(lons, lats, attribution, transform=ccrs.PlateCarree(), cmap='binary')

    # add features to map
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=.3, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=.3)
    ax.coastlines()
    ax.gridlines()

    plt.show()

    # save picture in fig_path
    plt.savefig(fig_path + fig_name + '.png')
    print('figure saved as ' + fig_name + '.png')


def check_args_plot_iml_regions(inputs, model, sign, outlier_perc, target, noise_tunnel,
                                fig_name, fig_path, path_to_data):
    """
    Check arguments of 'create_plot_label'
    """

    # checks inputs
    if not isinstance(inputs, (tuple, torch.Tensor)):
        sys.exit("`inputs` must be tuple or tensor.")

    # checks model
    if not isinstance(model, (torchvision.models.resnet.ResNet, CBAMResCNNEncoder)):
        sys.exit("`inputs` must be a torchvision.models.")

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
            "The number of samples provided in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )

    # checks noise_tunnel
    if not isinstance(noise_tunnel, bool):
        sys.exit("`noise_tunnel` must be a boolean.")

    # checks fig_name
    if not isinstance(fig_name, str):
        sys.exit("`fig_name` must be a String.")

    # checks fig_path
    if not isinstance(fig_path, str):
        sys.exit("`fig_path` must be a string.")

    if not os.path.exists(fig_path):
        sys.exit("`fig_path` does not exists.")

    # checks path_to_data
    check_args_path_to_data(path_to_data)


def check_args_path_to_data(path_to_data):
    """
    Checks path separately (reduce complexity)
    Args:
        path_to_data:

    Returns error, if it is not a path
    """
    if not isinstance(path_to_data, str):
        sys.exit("`path_to_data` must be a string.")

    if not path_to_data[-3:] == ".nc":
        sys.exit("Can only load netcdf files with file extension '.nc'.")

    if not os.path.exists(path_to_data):
        sys.exit("`path_to_data` does not exists.")