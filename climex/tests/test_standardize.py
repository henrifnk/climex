import unittest
import numpy as np
import xarray as xr
from climex.data.split_data import split_data
from climex.data.standardize import standardize


class TestStandardize(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.total_xr = xr.open_dataset(cls.path_to_data, engine='netcdf4')
        cls.split_dict = split_data(cls.total_xr, 5, 5, splitting_method='random', season='summer')
        cls.standard_dict = standardize(cls.split_dict)

    def test_output_format(self):
        """
        Tests if the output format is a dictionary of correct shape
        standardization should not change any shape or name within the dictionary
        """
        self.assertEqual(list(self.standard_dict.keys()),
                         ['train', 'val', 'test', 'train_labels', 'val_labels', 'test_labels']
                         )
        self.assertEqual(self.standard_dict.keys(),
                         self.split_dict.keys()
                         )
        for key in list(self.split_dict.keys()):
            self.assertEqual(
                self.standard_dict[key].shape,
                self.split_dict[key].shape
            )

    def test_untouched(self):
        """
        Assert that all labels remain untouched by standardizing
        """
        for key in ['train_labels', 'val_labels', 'test_labels']:
            self.assertTrue(
                np.allclose(
                    self.standard_dict[key],
                    self.split_dict[key]
                )
            )

    def test_norm(self):
        """
        Tests if the training set is  standardized
        """
        self.assertTrue(
            np.allclose(
                np.mean(self.standard_dict['train'], axis=(0, 2, 3)),
                np.array([0, 0])
            )
        )
        self.assertTrue(
            np.allclose(
                np.std(self.standard_dict['train'], axis=(0, 2, 3)),
                np.array([1., 1.])
            )
        )


if __name__ == '__main__':
    unittest.main()
