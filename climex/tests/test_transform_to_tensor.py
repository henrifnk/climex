import unittest
import xarray as xr
import numpy as np
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from climex.data.split_data import split_data
from climex.data.standardize import standardize
from climex.data.transform_to_tensor import transform_to_tensor
from climex.data.Dataset import get_Loader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class TestTransformToTensor(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        total_xr = xr.open_dataset(path_to_data, engine='netcdf4')
        split_dict = split_data(total_xr, 5, 5, splitting_method='sequential', season='winter')
        cls.standard_dict = standardize(split_dict)
        # initialize output
        cls.time_depth = 3
        cls.img_Tensor = transform_to_tensor(cls.standard_dict)
        cls.img_Tensor_not_shuffled = transform_to_tensor(cls.standard_dict, shuffle_train_data=False)
        cls.vid_Tensor = transform_to_tensor(cls.standard_dict, time_depth=cls.time_depth,
                                             target=-1, key='video')
        cls.svid_Tensor = transform_to_tensor(cls.standard_dict, time_depth=cls.time_depth,
                                              target=-1, key='sliding_video')
        cls.img_Loader = get_Loader(key='image', dta=cls.standard_dict['train'],
                                    lbl=cls.standard_dict['train_labels'],
                                    td=None, tgt=None,
                                    tf=transforms.Compose([transforms.ToTensor()]))
        cls.vid_Loader = get_Loader(key='video', dta=cls.standard_dict['test'],
                                    lbl=cls.standard_dict['test_labels'],
                                    td=cls.time_depth, tgt=-1,
                                    tf=transforms.Compose([transforms.ToTensor()]))
        cls.svid_Loader = get_Loader(key='sliding_video', dta=cls.standard_dict['train'],
                                     lbl=cls.standard_dict['test_labels'],
                                     td=cls.time_depth, tgt=-1,
                                     tf=transforms.Compose([transforms.ToTensor()]))

    def test_get_Loader(self):
        for key in ['image', 'video', 'sliding_video']:
            if key == 'image':
                Data = self.img_Loader
                img, label = Data.__getitem__(3)
                # assert an image that expands the grid over Europe
                self.assertEqual(list(img.data.shape), [2, 16, 39])
                self.assertIn(label, [0.0, 11.0, 17.0])
            elif key == 'video':
                Data = self.vid_Loader
                img, labels = Data.__getitem__(1)
                labels = labels[0]
                # assert an video with time_depth time stamps that expands the grid over Europe
                self.assertEqual(list(img.data.shape), [self.time_depth, 2, 16, 39])
            else:
                Data = self.svid_Loader
                img, label = Data.__getitem__(1)
                self.assertEqual(list(img.data.shape), [self.time_depth, 2, 16, 39])
                self.assertIn(label, [0.0, 11.0, 17.0])

            self.assertTrue(np.isin(label, [0., 11., 17.]).tolist())
            self.assertIsInstance(img, Tensor)
            self.assertIsInstance(Data, Dataset)

    def test_tensor_trafo(self):
        """
        Test transform_to tensor function for type of objects and consistency in methods.
        """
        for key in ['image', 'video', 'sliding_video']:
            # Iterate through all types of Data Loading ...
            for set in [0, 1, 2]:
                # ...and though train test an validation set
                length = len(list(self.standard_dict.values())[set + 3])
                if key == 'image':
                    Tensor = self.img_Tensor[set]
                elif key == 'video':
                    Tensor = self.vid_Tensor[set]
                    length = int(length / self.time_depth)
                else:
                    Tensor = self.svid_Tensor[set]
                    length = length - self.time_depth + 1

                self.assertIsInstance(Tensor, DataLoader)
                self.assertEqual(Tensor.__len__(), length)

    def test_shuffle(self):
        """
        Test that shuffle works correctly.
        """
        tensorTrainShuffled = self.img_Tensor[0]
        tensorTrainNotShuffled = self.img_Tensor_not_shuffled[0]

        self.assertIsInstance(tensorTrainShuffled.batch_sampler.sampler, RandomSampler)
        self.assertIsInstance(tensorTrainNotShuffled.batch_sampler.sampler, SequentialSampler)


if __name__ == '__main__':
    unittest.main()
