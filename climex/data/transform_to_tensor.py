import torch
from climex.data.Dataset import get_Loader
from torch.utils.data import DataLoader
from torchvision import transforms


def transform_to_tensor(dic, time_depth=None, target=None, key='image', batch_size=1,
                        transform=transforms.Compose([transforms.ToTensor()]), shuffle_train_data=True):
    """Loads the data in DataLoader from pytorch utils.

    Args:
        dic (dict): float64 array Data set might be train, validation or test data with shape
            [n_images, 2 (n_vars), 16 (lat), 19 (long)].
        key (str): defines the type and shape of the loaded data. Feasible arguments are
            'image', 'video' and 'sliding_video'.
        time_depth: (int): time_depth that defines the length of the video. Set to None if
            key = 'image'.
        target: (int): integer target that defines which picture in the video defines the label.
            Set to None if key is not 'sliding_video'.
        batch_size (int): passed to DataLoader, defines the number of samples that will be
            propagated through the network.
        transform: transformation methods passed to transforms.Compose defaults to ToTensor.
            Other methods to  augment can be added.
        shuffle_train_data (bool): determines whether the training data should be shuffled in each iteration.

    Returns:
        Train, test and validation Data Loader object that can be used during model training to
        load data sets in a feasible shape.
    """

    dataLoaders = {'train': None, 'test': None, 'val': None}
    for data_set in ['train', 'test', 'val']:
        label = data_set + '_labels'
        if data_set == 'train' and shuffle_train_data is True:
            shuffle = True
        else:
            shuffle = False

        dataClass = get_Loader(key=key, dta=dic[data_set],  lbl=dic[label],
                               td=time_depth, tgt=target, tf=transform)
        dataLoaders[data_set] = DataLoader(dataClass, batch_size=batch_size, shuffle=shuffle)

    return dataLoaders['train'], dataLoaders['test'], dataLoaders['val']
