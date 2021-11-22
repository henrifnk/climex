import unittest
import numpy as np
import xarray as xr
from docs.distance.calculate_distance import calculate


class TestDistanceCalculate(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.total_xr = xr.open_dataset(cls.path_to_data, engine='netcdf4')
        cls.input_label = list(cls.total_xr.variables['labels'][:])
        cls.input_mslp = cls.total_xr.variables['mslp'][:]
        cls.output_label, cls.output_data_index, cls.output_mslp = calculate(cls.input_label, cls.input_mslp)

    def test_output_len(self):
        """
        Test whether the size of the output, the number of elements is identical as expected
        """
        self.assertEqual(len(self.output_label), 2 * (self.input_label.count(11) + self.input_label.count(17)))
        self.assertEqual(len(self.output_data_index), len(self.output_label), len(self.output_mslp))

    def test_output_value(self):
        """
        Manually create data to check whether the output is correct
        """
        test_input_label = [0, 11, 11, 17]
        test_input_mslp = [1154, 6318, 255, 2262]
        correct_output_mslp = [[np.nan, 516, 90, 111], [np.nan, np.nan, 606, 406], [np.nan, np.nan, np.nan, 201],
                               [np.nan, np.nan, np.nan, np.nan]]

        output_label, output_data_index, output_mslp = calculate(test_input_label, test_input_mslp)
        output_mslp = np.array(list(output_mslp), dtype=object).tolist()

        self.assertEqual(correct_output_mslp, output_mslp)


if __name__ == '__main__':
    unittest.main()
