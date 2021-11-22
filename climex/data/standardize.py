import numpy as np


def standardize(dict_recoded):
    """Standardize dictionary by information from training set per channel.
    Args:
        dict_recoded (dict): A dictionary containing the numpy arrays with recoded labels from
            recode_labels:
            {
                'train': float64 array with shape [n_train_images, 2 (n_vars), 16 (lat), 19 (long)]
                'val': float64 array with shape [n_val_images, 2 (n_vars), 16 (lat), 19 (long)]
                'test': float64 array with shape [n_test_images, 2 (n_vars), 16 (lat), 19 (long)]
                'train_labels': float64 array with shape [n_train_images, ]
                'val_labels': float64 array with shape [n_val_images, ]
                'test_labels': float64 array with shape [n_test_images, ]
            }

        Returns:
            The dictionary containing the numpy arrays see 'Args', where the data sets are
            standardized by training data information.
    """

    mean_ = np.mean(dict_recoded['train'], axis=(0, 2, 3))
    std_ = np.std(dict_recoded['train'], axis=(0, 2, 3))
    dict_standard = dict_recoded
    for data_set in ["train", "val", "test"]:
        tmp = dict_recoded[data_set]
        tmp[:, 0, :, :] = (tmp[:, 0, :, :] - mean_[0]) / std_[0]
        tmp[:, 1, :, :] = (tmp[:, 1, :, :] - mean_[1]) / std_[1]
        dict_standard[data_set] = tmp

    return dict_standard
