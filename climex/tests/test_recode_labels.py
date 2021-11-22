import unittest
import numpy as np
import xarray as xr
from climex.data.split_data import split_data
from climex.data.recode_labels import recode_labels


class TestRecodeLabels(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.total_xr = xr.open_dataset(cls.path_to_data, engine='netcdf4')
        cls.split_dict = split_data(cls.total_xr, 5, 5, splitting_method='random', season='summer')
        cls.dict_recoded = recode_labels(cls.split_dict)

    def test_output_format(self):
        """
        Tests if the output format is a dictionary of correct shape
        standardization should not change any shape or name within the dictionary.
        """
        self.assertEqual(self.dict_recoded.keys(), self.split_dict.keys())

        for key in list(self.split_dict.keys()):
            self.assertEqual(
                self.dict_recoded[key].shape,
                self.split_dict[key].shape
            )

    def test_untouched(self):
        """
        Assert that all datasets remain untouched by standardizing.
        """
        for key in ['train', 'val', 'test']:
            self.assertTrue(
                np.allclose(
                    self.dict_recoded[key],
                    self.split_dict[key]
                )
            )

    def test_recoded(self):
        """
        Tests if the labels are recoded.
        """
        dict_recoded = self.dict_recoded
        for key in ['train_labels', 'val_labels', 'test_labels']:
            label_is_correct = np.isin(dict_recoded[key], [0, 1, 2])
            self.assertTrue(all(label_is_correct))


if __name__ == '__main__':
    unittest.main()
