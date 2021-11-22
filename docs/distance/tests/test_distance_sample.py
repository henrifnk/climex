import unittest
import xarray as xr
from docs.distance.sample import sample


class TestDistanceSample(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.total_xr = xr.open_dataset(cls.path_to_data, engine='netcdf4')
        cls.input_label = list(cls.total_xr.variables['labels'][:])
        cls.input_mslp = cls.total_xr.variables['mslp'][:]
        cls.output_label, cls.output_data_index, cls.output_mslp = sample(cls.input_label, cls.input_mslp)

    def test_output_len(self):
        """
        Tests if the output length of the dataset is identical and
        also whether output length is identical with input length
        if in sample.py uses
        #sample_0=random.sample(data_index_0, len(data_index_11)+len(data_index_17))
        then should test
        # self.assertEqual(len(self.output_label),2*(self.input_label.count(11)+self.input_label.count(17)))
        """

        # self.assertEqual(len(self.output_label), len(self.input_label))
        self.assertEqual(len(self.output_label), 2 * (self.input_label.count(11) + self.input_label.count(17)))
        self.assertEqual(len(self.output_label), len(self.output_mslp), len(self.output_data_index))

    def test_output_value(self):
        """
        Manually create data to check whether the output is correct
        """
        test_input_label = [0, 11, 11, 0, 17]
        test_input_mslp = [1154, 6318, 255, 2262, 555]

        correct_output_label = [0, 0, 11, 11, 17]
        correct_output_index = [0, 3, 1, 2, 4]
        correct_output_mslp = [1154, 2262, 6318, 255, 555]

        output_label, output_data_index, output_mslp = sample(test_input_label, test_input_mslp)
        self.assertEqual(correct_output_index, output_data_index)
        self.assertEqual(correct_output_label, output_label)
        self.assertEqual(correct_output_mslp, output_mslp)


if __name__ == '__main__':
    unittest.main()
