import torch
import numpy as np
from torch.utils.data import Dataset
from climex.models.utils import add_gnoise
from collections import Counter


def get_Loader(key, dta, lbl, td, tgt, tf):
    """Loads the data in a Dataset child class.

    Args:
        key (str): Defines the type and shape of the loaded data. Feasible arguments are
            'image', 'video' and 'sliding_video'.
        dta (float64 array): Data set might be train, validation or test data with shape
            [n_images, 2 (n_vars), 16 (lat), 19 (long)].
        lbl (float64 array): Corresponding label to data with shape [n_images, ]
        td (int): time_depth that defines the length of the video. Set to None if
            key = 'image'.
        tgt (int): integer target that defines which picture in the video defines the label.
            Set to None if key is not 'sliding_video'.
        tf: transformation methods passed to transforms.Compose.

    Returns:
        Dataset object that can be passed to DataLoader function.
    """

    # Load Image (CNN)------------------------------------------------------------------------------
    if key == 'image':
        class ImageDataset(Dataset):
            """Image data set for pytorch usage.

            Attributes:
                data (float64 array): array with shape [n_images, 2 (n_vars), 16 (lat), 19 (long)]
                label (float64 array):  float64 array with shape [n_images, ]
                transform (callable, optional): Optional transform to be applied on a sample.
                    Default is None.
            """
            def __init__(self, data, label, transform=None):
                """Loads the data in images shape to be utilized by pytorch.
                """

                self.data = data
                self.label = label
                self.transform = transform

            def __getitem__(self, index):
                """Get (image data) items by index during model training.
                """

                img = self.data[index]
                img = np.transpose(img, (1, 2, 0))
                label = self.label[index]
                if self.transform is not None:
                    img = self.transform(img)
                return img, label

            def __len__(self):
                """Get (image data) length.
                """

                return len(self.data)

            def get_label_weight(self):
                """Calculate (image data) class label weights by label frequency.
                Should only be called on the training set.
                """

                weights_ = Counter(self.label)
                weights = [1 - (x / sum(weights_.values())) for x in weights_.values()]
                return weights

        return ImageDataset(dta, lbl, tf)

    # Load Video (LSTM)-----------------------------------------------------------------------------
    if key == 'video':
        class VideoDataset(Dataset):
            """Video data set for pytorch usage.

            Attributes:
                data (float64 array): array with shape [n_images, 2 (n_vars), 16 (lat), 19 (long)]
                label (float64 array):  float64 array with shape [n_images, ]
                transform (callable, optional): Optional transform to be applied on a sample.
                    Default is None.
                time_depth (int): int of length 1 indicating video length.
            """

            def __init__(self, data, label, time_depth, transform=None):
                """Loads the data in video shape to be utilized by pytorch.
                """

                # convert from image to video format
                # cut the last timestamps
                data = data[:int(data.shape[0] / time_depth) * time_depth, :, :, :]
                video = data.reshape(int(data.shape[0] / time_depth), time_depth,
                                     data.shape[1], data.shape[2], data.shape[3])
                label = label[:int(label.shape[0] / time_depth) * time_depth]
                label_video = label.reshape(int(label.shape[0] / time_depth), time_depth)

                self.data = video
                self.label = label_video
                self.transform = transform

            def __getitem__(self, index):
                """Get (video data) items by index during model training.
                """

                video = self.data[index]
                label = self.label[index]
                if self.transform is not None:
                    frames_tr = []
                    for frame in video:
                        frame = np.transpose(frame, (1, 2, 0))
                        frame = self.transform(frame)
                        frames_tr.append(frame)
                    video = torch.stack(frames_tr)
                return video, label

            def __len__(self):
                """Get (video data) length.
                """
                return len(self.data)

            def get_label_weight(self):
                """Calculate (video data) class label weights by label frequency.
                Should only be called on the training set.
                """
                weights_ = Counter(self.label)
                weights = [1 - (x / sum(weights_.values())) for x in weights_.values()]
                return weights

        return VideoDataset(dta, lbl, td, tf)

    # Load sliding Video (LSTM)---------------------------------------------------------------------
    if key == 'sliding_video':
        class SlidingVideoDataset(Dataset):
            """Sliding video data set for pytorch usage.

            Attributes:
                    data (float64 array): array with shape [n_images, 2 (n_vars), 16 (lat), 19 (long)]
                    label (float64 array):  float64 array with shape [n_images, ]
                    transform (callable, optional): Optional transform to be applied on a sample.
                        Default is None.
                    time_depth (int): int of length 1 indicating video length.
                    target (int): Indicating which picture of the video serves for the
                            prediction label.
            """
            def __init__(self, data, label, time_depth, target, transform=None):
                """Initializes the data in sliding video shape to be utilized by pytorch.
                """
                self.data = data
                self.label = label
                self.time_depth = time_depth
                self.transform = transform
                self.target = target

            def __getitem__(self, index):
                """Get (sliding video data) items by index during model training.
                """
                video = self.data[index:index + self.time_depth]
                video = add_gnoise(video)
                label = self.label[index:index + self.time_depth][self.target - 1]
                if self.transform is not None:
                    frames_tr = []
                    for frame in video:
                        frame = np.transpose(frame, (1, 2, 0))
                        frame = self.transform(frame)
                        frames_tr.append(frame)
                    video = torch.stack(frames_tr)
                return video, label

            def __len__(self):
                """Get (sliding video data) data length.
                """
                return len(self.data) - self.time_depth + 1

            def get_label_weight(self):
                """Calculate (sliding video data) class label weights by label frequency.
                Should only be called on the training set.
                """
                weights_ = Counter(self.label)
                weights = [1 - (x / sum(weights_.values())) for x in weights_.values()]
                return weights

        return SlidingVideoDataset(dta, lbl, td, tgt, tf)
