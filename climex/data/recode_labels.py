def recode_labels(data_dict):
    """Recode labels from (0, 11, 17) to (0, 1, 2) as required by pytorch.

    Args:
        data_dict (dict): a dictionary containing the numpy arrays from split data:
            {
                'train': float64 array with shape [n_train_images, 2 (n_vars), 16 (lat), 19 (long)]
                'val': float64 array with shape [n_val_images, 2 (n_vars), 16 (lat), 19 (long)]
                'test': float64 array with shape [n_test_images, 2 (n_vars), 16 (lat), 19 (long)]
                'train_labels': float64 array with shape [n_train_images, ]
                'val_labels': float64 array with shape [n_val_images, ]
                'test_labels': float64 array with shape [n_test_images, ]
            }

        Returns:
            A dictionary containing the numpy arrays see 'Args', where the labels are recoded.
    """

    dict_recoded = data_dict
    for lbl in ["train_labels", "val_labels", "test_labels"]:
        tmp_label = data_dict[lbl]
        tmp_label[tmp_label == 11] = int(1)
        tmp_label[tmp_label == 17] = int(2)
        dict_recoded[set] = tmp_label

    return dict_recoded
