import unittest
import xarray as xr
from docs.insights.input import get_input_data


class TestInput(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'

        cls.total_xr = xr.open_dataset(cls.path_to_data, engine='netcdf4')
        cls.lon = cls.total_xr['lon'].shape[0]
        cls.lat = cls.total_xr['lat'].shape[0]
        cls.time = cls.total_xr['z500'].shape[0]
        # number of variables: mslp & z500
        cls.variables = 2

    # Tests ---------------------------------------------------------------------------------------
    def test_get_input_shape_data(self):
        # get input data
        input_data = get_input_data(path_to_data=self.path_to_data, sample_size=1)

        # separate tuple
        variables, labels = input_data

        # check shape: variables must have 4 dimensions
        # [number of sampled inputs, number of variables, latitude, longitude]
        self.assertEqual(variables.dim(), 4)
        self.assertLessEqual(variables.shape[0], self.time)
        self.assertEqual(variables.shape[1], 2)
        self.assertEqual(variables.shape[2], self.lat)
        self.assertEqual(variables.shape[3], self.lon)

        # check shape: labels should have one dimension which is equal to the first dimension in the variables
        self.assertEqual(labels.dim(), 1)
        self.assertEqual(labels.shape[0], variables.shape[0])

    def test_get_input_data_shape_small(self):
        # get input data
        input_data = get_input_data(path_to_data=self.path_to_data, sample_size=0.1)

        # separate tuple
        variables, labels = input_data

        # check shape: variables must have 4 dimensions
        # [number of sampled inputs, number of variables, latitude, longitude]
        self.assertEqual(variables.dim(), 4)
        self.assertLessEqual(variables.shape[0], self.time)
        self.assertEqual(variables.shape[1], self.variables)
        self.assertEqual(variables.shape[2], self.lat)
        self.assertEqual(variables.shape[3], self.lon)

        # check shape: labels should have one dimension which is equal to the first dimension in the variables
        self.assertEqual(labels.dim(), 1)
        self.assertLessEqual(labels.shape[0], self.time)

    def test_get_input_data_values(self):
        # get input data
        input_data = get_input_data(path_to_data=self.path_to_data, sample_size=1)

        # separate tuple
        variables, labels = input_data

        # labels must be either 0,1 or 2
        values_labels = labels.unique(dim=0).cpu().detach().numpy()
        for label in values_labels:
            self.assertIn(label, (0, 1, 2))


if __name__ == '__main__':
    unittest.main()