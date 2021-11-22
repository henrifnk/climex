import unittest
import xarray as xr

from climex.data.split_data import (split_data, split_train_val_test_seq, split_train_val_test_rand,
                                    subset_by_season)


class TestSplitTrainValTest(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """Inits TestSplitTrainValTest with meta info and by loading in unit test data."""

        path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'

        # Set test specifications
        cls.total_xr = xr.open_dataset(path_to_data, engine='netcdf4')
        cls.total_df = cls.total_xr.to_dataframe()
        # All timestamps of our unit test data
        cls.obs_times = cls.total_df.index.get_level_values('time')
        # Number of observation years in our unit test data
        cls.n_obs_years = len(cls.obs_times.year.unique())
        # Number of images that we have in each year per year (same for all years)
        cls.images_per_year = cls.get_images_per_year(cls.obs_times)

    # Helper methods -------------------------------------------------------------------------------
    def get_images_per_year(timestamps):
        """Get the number of images that we have in each year per year (same for all years)

            Args:
                timestamps: DatetimeIndex with observation timestamps.

            Returns:
            Number of images as integer
        """

        # Number of included observation months per year in our unit test data
        n_obs_months_per_year = len(timestamps.month.unique())
        # Number of included observation days per month in our unit test data
        n_obs_days_per_month = len(timestamps.day.unique())
        # Number of included observation days per year in our unit test data
        n_obs_days_per_year = int(n_obs_days_per_month * n_obs_months_per_year)

        return n_obs_days_per_year

    # Tests ---------------------------------------------------------------------------------------
    def test_final_output_formats(self):
        """ Tests if the `split_data` output format and dimensions are correct."""
        # All combinations of n_years_val and n_years_test we want to test based on
        all_n_years_val = [5, 8]
        all_n_years_test = [5, 10]
        arg_years = ((v, t) for v in all_n_years_val for t in all_n_years_test)
        # All combinations of methods and season
        all_methods_args = ['sequential', 'random']
        all_season_args = [None, 'winter', 'summer']
        arg_options = ((m, s) for m in all_methods_args for s in all_season_args)

        # For all argument combinations run tests
        for n_years_val, n_years_test in arg_years:
            for method_arg, season_arg in arg_options:

                split_dict = split_data(
                    self.total_xr, n_years_val, n_years_test,
                    splitting_method=method_arg, season=season_arg
                )

                # Check whether output dict is of correct length
                self.assertEqual(len(split_dict), 6)
                n_train_years = self.n_obs_years - n_years_val - n_years_test
                images_per_year = self.images_per_year
                # If splitted by season we only have half the total images each year
                if season_arg is not None:
                    images_per_year = images_per_year / 2

                exp_n_images_train = n_train_years * images_per_year
                exp_n_images_val = n_years_val * images_per_year
                exp_n_images_test = n_years_test * images_per_year
                # Check whether training, val and test sets have correct dimensions
                self.assertEqual(split_dict['train'].shape, (exp_n_images_train, 2, 16, 39))
                self.assertEqual(split_dict['val'].shape, (exp_n_images_val, 2, 16, 39))
                self.assertEqual(split_dict['test'].shape, (exp_n_images_test, 2, 16, 39))
                self.assertEqual(split_dict['train_labels'].shape, (exp_n_images_train,))
                self.assertEqual(split_dict['val_labels'].shape, (exp_n_images_val,))
                self.assertEqual(split_dict['test_labels'].shape, (exp_n_images_test,))

    def test_correct_split_seq(self):
        """Tests that the observation years are assigned correctly to train, val and test sets
            for method 'sequential'.
        """
        # Min and max years of our unit test data set
        max_year = self.obs_times.year.max()
        min_year = self.obs_times.year.min()

        all_n_years_val = [5, 8]
        all_n_years_test = [5, 10]
        arg_years = ((v, t) for v in all_n_years_val for t in all_n_years_test)

        # For all combinations of specified n_years_val and n_years_test run tests
        for n_years_val, n_years_test in arg_years:
            # Expected years for train, validation and test sets
            exp_years_test = list(range(max_year - n_years_test + 1, max_year + 1))
            exp_years_val = list(range(max_year - n_years_test - n_years_val + 1, min(exp_years_test)))
            exp_years_train = list(range(min_year, min(exp_years_val)))

            # Split observation years sequentially
            split_dict = split_train_val_test_seq(self.total_xr, n_years_val, n_years_test)
            train_set = split_dict["train"].to_dataframe()
            val_set = split_dict["val"].to_dataframe()
            test_set = split_dict["test"].to_dataframe()
            years_train = train_set.index.get_level_values('time').year.unique()
            years_val = val_set.index.get_level_values('time').year.unique()
            years_test = test_set.index.get_level_values('time').year.unique()

            # Check whether years are correctly assigned to train, validation and test sets
            self.assertEqual(len(years_train), self.n_obs_years - n_years_val - n_years_test)
            self.assertEqual(years_train.tolist(), exp_years_train)
            self.assertEqual(len(years_val), n_years_val)
            self.assertEqual(years_val.tolist(), exp_years_val)
            self.assertEqual(len(years_test), n_years_test)
            self.assertEqual(years_test.tolist(), exp_years_test)

    def test_correct_split_rand(self):
        """Tests that the observation years are assigned correctly to train, val and test sets
            for method 'random'.
        """
        all_n_years_val = [5, 8]
        all_n_years_test = [5, 10]
        arg_years = ((v, t) for v in all_n_years_val for t in all_n_years_test)

        # For all combinations of specified n_years_val and n_years_test run tests
        for n_years_val, n_years_test in arg_years:
            n_years_train = self.n_obs_years - n_years_val - n_years_test
            split_dict = split_train_val_test_rand(self.total_xr, n_years_val, n_years_test)

            # Train
            train_set = split_dict["train"].to_dataframe()
            years_train = train_set.index.get_level_values('time').year.unique()
            self.assertEqual(len(years_train), n_years_train)

            # Val
            val_set = split_dict["val"].to_dataframe()
            years_val = val_set.index.get_level_values('time').year.unique()
            self.assertEqual(len(years_val), n_years_val)

            # Test
            test_set = split_dict["test"].to_dataframe()
            years_test = test_set.index.get_level_values('time').year.unique()
            self.assertEqual(len(years_test), n_years_test)

            # Check that there is no intersection
            self.assertEqual(list(set(years_train) & set(years_val)), [])
            self.assertEqual(list(set(years_train) & set(years_test)), [])
            self.assertEqual(list(set(years_val) & set(years_test)), [])

    def test_correct_season_filtering(self):
        """Tests that filtering per season works correctly."""

        # Check that winter and summer xarrays contain the right months
        winter_xr = subset_by_season(self.total_xr, season='winter')
        all(winter_xr['time.month'].to_dataframe()['month'].unique() == [1, 2, 3, 10, 11, 12])
        summer_xr = subset_by_season(self.total_xr, season='winter')
        all(summer_xr['time.month'].to_dataframe()['month'].unique() == [4, 5, 6, 7, 8, 9])

        # Check that no data got lost
        rows_winter = winter_xr.to_dataframe().shape[0]
        rows_summer = summer_xr.to_dataframe().shape[0]
        rows_total = self.total_xr.to_dataframe().shape[0]
        self.assertEqual(rows_winter + rows_summer, rows_total)


if __name__ == '__main__':
    unittest.main()
