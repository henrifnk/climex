from climex.data.split_data_helpers import (split_train_val_test_seq, split_train_val_test_rand,
                                            subset_by_season, xarray_dict_to_np_dict)


def split_data(my_xarray, n_years_val, n_years_test, splitting_method='sequential', season=None, save_test=None):
    """Splits observation years into train, validation and test sets.
    Args:
        my_xarray (array): xarray containing read in weather data.
        n_years_val (int): the number of years to include in the validation set.
        n_years_test (int): the number of years to include in the test set.
        splitting_method (str): the splitting method executed. Must be 'sequential' or 'random'.
        season (str, optional): setting it to None means that all images will be loaded.
            Alternatively, it can be set to 'winter' or 'summer' to only load images
            from winter or summer months respectively.
            Default is None.
        save_test (str): location where to store test set
            Set to None where no saving is nessecary.

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

    # Filters my_xarray by season, if required.
    if season is not None:
        my_xarray = subset_by_season(my_xarray, season=season)

    # Assign years sequentially or randomly to train, validation and test sets.
    if splitting_method == 'sequential':
        xarray_split_dict = split_train_val_test_seq(my_xarray, n_years_val, n_years_test, save_test=save_test)
    else:
        xarray_split_dict = split_train_val_test_rand(my_xarray, n_years_val, n_years_test, save_test=save_test)

    # Transform xarrays to numpy arrays
    np_split_dict = xarray_dict_to_np_dict(xarray_split_dict)

    return np_split_dict
