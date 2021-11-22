import numpy as np
import logging


def split_train_val_test_seq(my_xarray, n_years_val, n_years_test, save_test=None):
    """Sequentially splits observation years into train, validation and test sets.
        The split is temporarily ordered as follows:
        train set | validation set | test set.

    This means that the images of the:
        - first "(110 - n_years_val - n_years_test)" years are assigned to the training set
        - following "n_years_val" years are assigned to the validation set
        - last "n_years_test" years are assigned to the test set

    Args:
        my_xarray (array): xarray containing read in weather data.
        n_years_val (int): the number of years to include in the validation set.
        n_years_test (int): the number of years to include in the test set.
        save_test (str): location where to store test set
            Set to None where no saving is nessecary.


    Returns:
        A dict containing train, validation and test sets.
            {
                'Training': xarray containing "(110 - n_years_val - n_years_test)" years
                'Validation': xarray containing "n_years_val" years
                'Test': xarray containing "n_years_test" years
            }
    """

    # Extract max date and year from total data set
    max_date = my_xarray['time'].max().values
    max_year = my_xarray['time.year'].max().values

    # Create test set
    first_year_test = max_year - n_years_test + 1
    first_date_test = '01.01.' + str(first_year_test)
    test_xr = my_xarray.sel(time=slice(first_date_test, max_date))

    if save_test is not None:
        logging.basicConfig(filename=save_test + 'logging.log', level=logging.INFO, format='%(message)s')
        logging.getLogger(__name__)
        logging.info('Test set starts at: ' + first_date_test + 'test set ends after ' + str(n_years_test) + 'years')
    # Create val set
    first_year_val = first_year_test - n_years_val
    first_date_val = '01.01.' + str(first_year_val)
    last_date_val = '31.12.' + str(first_year_test - 1)
    val_xr = my_xarray.sel(time=slice(first_date_val, last_date_val))



    # Create test set
    last_date_test = '31.12.' + str(first_year_val - 1)
    train_xr = my_xarray.sel(time=slice(None, last_date_test))

    split_dict = {
        'train': train_xr,
        'val': val_xr,
        'test': test_xr
    }

    return split_dict


def split_train_val_test_rand(my_xarray, n_years_val, n_years_test, seed=45476857, save_test=None):
    """Randomly splits observation years into train, validation and test sets.

    Randomly assigns 'n_years_val' years to the validation set and 'n_years_test'
    to the test set without intersection. The remaining years are assigned to the
    training set.
    Then all images of the corresponding years form train, validation and test sets
    respectively.

    Args:
        my_xarray (array): xarray containing read in weather data.
        n_years_val (int): the number of years to include in the validation set.
        n_years_test (int): the number of years to include in the test set.
        seed (int): set seed with np.random
        save_test (str): location where to store test set
            Set to None where no saving is nessecary.

    Returns:
        A dict containing train, validation and test sets.
        {
            'Training': xarray containing "(110 - n_years_val - n_years_test)" years
            'Validation': xarray containing "n_years_val" years
            'Test': xarray containing "n_years_test" years
        }
    """

    np.random.seed(seed)

    # Extract max and min year from total data set
    min_year = my_xarray['time.year'].min().values
    max_year = my_xarray['time.year'].max().values

    # Randomly sample years for test set (without replacement) from all years
    all_years = np.arange(min_year, max_year + 1)
    test_years = np.random.choice(all_years, size=n_years_test, replace=False)

    # Randomly sample years for val and test sets (without replacement) from remaining years
    all_val_train_years = all_years[~np.in1d(all_years, test_years)]
    val_years = np.random.choice(all_val_train_years, size=n_years_val, replace=False)
    train_years = all_val_train_years[~np.in1d(all_val_train_years, val_years)]

    # Create train, val and test xarrays
    test_xr = my_xarray.sel(time=my_xarray['time.year'].isin(test_years))
    if save_test is not None:
        logging.basicConfig(filename=save_test + 'logging.log', level=logging.INFO, format='%(message)s')
        logging.getLogger(__name__)
        logging.info('Random test years: ' + str(test_years))
    val_xr = my_xarray.sel(time=my_xarray['time.year'].isin(val_years))
    train_xr = my_xarray.sel(time=my_xarray['time.year'].isin(train_years))

    split_dict = {
        'train': train_xr,
        'val': val_xr,
        'test': test_xr
    }

    return split_dict


def subset_by_season(my_xarray, season, winter_months=[10, 11, 12, 1, 2, 3]):
    """Filters my_xarray by the months of the required season.

    Args:
        my_xarray (array): xarray containing read in weather data to be filtered by season.
        season (str, optional): setting it to None means that all images will be loaded.
            Alternatively, it can be set to 'winter' or 'summer' to only load images
            from winter or summer months respectively.
            Default is None.
        winter_months (list): The month numbers defining the winter season.

    Returns:
        A xarray filtered by season.
    """

    if season == 'winter':
        req_months = winter_months
    else:
        req_months = np.setdiff1d(np.array(range(0, 13)), winter_months)

    # Subset xarray for winter or summer months
    season_xr = my_xarray.sel(time=my_xarray['time.month'].isin(req_months))

    return season_xr


def transform_to_np_array(my_xarray):
    """Transforms an xarray into two numpy arrays containing the images and the labels.

    The image array has shape [2 (n_vars), n_images, 16 (lats), 39 (lons)] and the labels array
    has shape [n_images, ].
    n_vars is the number of variables / channels, n_images is the number of images contained in
    the xarray and lats and lons the number of latitudes and number of longitudes respectively.

    Args:
        my_xarray (array): xarray containing read in weather data.

    Returns:
        - The image numpy array of shape [2, n_images, 16, 19]
        - and the label numpy array of shape [n_images, ]
    """

    # Convert xarray to float array of shape [n_vars, n_images, lats, lons]
    images = my_xarray[['z500', 'mslp']].to_array().values
    # Change dimensions to [n_images, n_vars, lats, lons] as required by pytorch
    images = np.swapaxes(images, 0, 1)
    # Image labels
    labels = my_xarray['labels'].values

    return images, labels


def xarray_dict_to_np_dict(xarray_dict):
    """Transforms each xarray in the `xarray_dict` dictionary into two numpy arrays containing
    the images and the labels.

    See documentation of `transform_to_np_array` method for further details

    Args:
        xarray_dict: The dictionary containing the xarrays.

    Returns:
        A dictionary containing the numpy arrays:
        {
            'train': float64 array with shape [n_train_images, 2 (n_vars), 16 (lat), 19 (long)]
            'val': float64 array with shape [n_val_images, 2 (n_vars), 16 (lat), 19 (long)]
            'test': float64 array with shape [n_test_images, 2 (n_vars), 16 (lat), 19 (long)]
            'train_labels': float64 array with shape [n_train_images, ]
            'val_labels': float64 array with shape [n_val_images, ]
            'test_labels': float64 array with shape [n_test_images, ]
        }
    """
    train, train_labels = transform_to_np_array(xarray_dict['train'])
    val, val_labels = transform_to_np_array(xarray_dict['val'])
    test, test_labels = transform_to_np_array(xarray_dict['test'])

    output_dict = {
        'train': train,
        'val': val,
        'test': test,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels
    }

    return output_dict
