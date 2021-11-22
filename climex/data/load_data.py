from climex.data.split_data import split_data
from climex.data.standardize import standardize
from climex.data.transform_to_tensor import transform_to_tensor
from climex.data.recode_labels import recode_labels
import xarray as xr
import sys
import os


def load_data(key='image', time_depth=None, target=None, batch_size=1,
              n_years_val=10, n_years_test=10, splitting_method='sequential', season=None, shuffle_train_data=True,
              path_to_data='climex/data/small_trainingset/training_database_daily.nc', save_test=None):
    """Loads the (splitted and processed) data in DataLoader objects from pytorch utils.

    The function serves as wrapper for the data pipeline that:
        - Ingests the raw netcdf files
        - Splits the data into train, validation and test sets
        - If required, subsets the data by season ('winter' or 'summer')
        - Standardizes the values of train, validation and test sets
        - Converts the data into an appropriate format for the pytorch framework

    The raw images from the netcdf file can be returned as images (pytorch class: ImageDataset),
    videos (pytorch class: VideoDataset) or sliding videos (pytorch class: SlidingVideoDataset).

    More information about the possible output classes (ImageDataset, VideoDataset and
    SlidingVideoDataset) of this function can be found in the pytorch documentation.

    Args:
        key (str): defines the type and shape of the loaded data. Feasible arguments are
            'image', 'video' and 'sliding_video'.
        time_depth: (int): time_depth that defines the length of the video. Set to None if
            key = 'image'.
        target: (int): integer target that defines which picture in the video defines the label.
            Set to None if key is not 'sliding_video'.
        batch_size (int): passed to DataLoader, defines the number of samples that will be
            propagated through the network.
        n_years_val (int): the number of years to include in the validation set.
        n_years_test (int): the number of years to include in the test set.
        splitting_method (str): the splitting method excecuted. Must be 'sequential' or 'random'.
        season (str, optional): setting it to None means that all images will be loaded.
            Alternatively, it can be set to 'winter' or 'summer' to only load images
            from winter or sommer months respectively.
        shuffle_train_data (bool): determines whether the training data should be suffled in each iteration.
            True by default.
        save_test (str): location where to store test set
            Set to None where no saving is nessecary.
        path_to_data (str): Path to the nc file with the data to be loaded.

        Returns:
            Train, test and validation Data Loader objects (from pytorch) that can be used during
            model training to load data sets in an appropriate shape.

        Examples:
            # Loads winter images only. Output class is ImageDataset.
            >>> load_data(key = 'image', season = 'winter')

            # Assign the images of random 10 years to the validation set and random 8 years to
            # the test set (without intersection). Output class is ImageDataset.
            >>> load_data(key = 'image', n_years_val = 10, n_years_test = 8,
            >>>           splitting_method = 'random')

            # Loads data as videos consisting of 10 images, where the class label of the 10th images
            # defines the class label of the video. Output class is VideoDataset.
            >>> load_data(key = 'video', time_depth = 10, target = 10)

            # Loads data as sliding videos consisting of 10 images, where the class label of the
            # 10th images defines the class label of the video. Output class is SlidingVideoDataset.
            >>> load_data(key = 'video', time_depth = 10, target = 10)
    """

    # Arguments checks
    check_args_load_data(key, time_depth, target, batch_size, n_years_val, n_years_test,
                         splitting_method, season, shuffle_train_data, path_to_data)

    # Data Ingestion
    total_xr = xr.open_dataset(path_to_data, engine='netcdf4')

    # Data Splitting
    dict_splitted = split_data(total_xr, n_years_val, n_years_test, splitting_method, season, save_test=save_test)

    # Data Preprocessing
    dict_splitted = recode_labels(dict_splitted)
    dict_splitted_std = standardize(dict_splitted)
    data_loaded = transform_to_tensor(dict_splitted_std, time_depth, target, key, batch_size, shuffle_train_data=shuffle_train_data)

    return data_loaded


def check_args_load_data(key, time_depth, target, batch_size, n_years_val, n_years_test,
                         splitting_method, season, shuffle_train_data, path_to_data):
    """Argument checks for `load_data` function. See `?load_data`.

    Raises:
        System Exit if argument checks are satisfied.
    """

    # Checks for key
    if key not in ['image', 'video', 'sliding_video']:
        sys.exit("`key` must be 'image', 'video' or 'sliding_video'.")

    # Checks for time_depth and target
    if key == 'image':
        if time_depth is not None:
            print("`time_depth` is set to None for 'image' key.")
        if target is not None:
            print("`target` is set to None for 'image' key.")
    elif key == "sliding_video":
        if not isinstance(target, int):
            sys.exit("`target` must be an integer for `key`s other than 'image'.")
        if abs(target) > time_depth:
            sys.exit("`target` cannot be greater than `time_depth`.")
    else:
        if not isinstance(time_depth, int):
            sys.exit("`time_depth` must be an integer for `key`s other than 'image'.")
        if time_depth <= 0:
            sys.exit("`time_depth` must be greater than 0.")

    # Checks for batch_size
    if not isinstance(batch_size, int):
        sys.exit("`batch_size` must be an integer.")

    if batch_size <= 0:
        sys.exit("`batch_size` must be greater than 0.")

    # Checks for n_years_val and n_years_test
    if not isinstance(n_years_val, int) or not isinstance(n_years_test, int):
        sys.exit("`n_years_val` and `n_years_test` must be integer.")

    if n_years_val <= 0 or n_years_test <= 0:
        sys.exit("`n_years_val` and `n_years_test` must be greater than 0.")

    # Checks for splitting_method
    if splitting_method not in ['sequential', 'random']:
        sys.exit("`splitting_method` must be 'sequential' or 'random'.")

    # Checks for season
    if season not in [None, 'summer', 'winter']:
        sys.exit("`season` must be None, 'winter' or 'summer'.")

    # Checks for shuffle_train_data
    if not isinstance(shuffle_train_data, bool):
        sys.exit("`shuffle_train_data` must be a boolean.")

    # Checks for path_to_data
    if not isinstance(path_to_data, str):
        sys.exit("`path_to_data` must be a string.")

    if not path_to_data[-3:] == ".nc":
        sys.exit("Can only load netcdf files with file extension '.nc'.")

    if not os.path.exists(path_to_data):
        sys.exit("`path_to_data` does not exists.")
