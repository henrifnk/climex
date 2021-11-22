# -*- coding: utf-8 -*-
# Note that this requires the full data set and therefore cannot be executed by the CI as we don't
# want to make our data public.
import xarray as xr
import random


def generate_data_for_unit_tests(path_raw_data='../../data/small_trainingset/training_database_daily.nc',
                                 path_unit_test_data='../testdata/training_database_daily_unit_tests.nc'):
    """Generates a netcdf data file for our unit tests.

       It takes a small subsample of our actual data (4 images per year) and replaces the actual
       values of our 'mslp' and 'z500' with dummy values to comply with data privacy concerns.

       Args:
        path_raw_data: Path to our netcdf files.
        path_unit_test_data: Path for where generated test data are saved
    """

    # Open netcdf as xarray
    total_xr = xr.open_dataset(path_raw_data, engine='netcdf4')

    # For our unit tests data only include the following months and days
    included_months = [1, 6]
    included_days = [1, 28]
    sub_xr = total_xr.sel(time=total_xr['time.month'].isin(included_months))
    sub_xr = sub_xr.sel(time=sub_xr['time.day'].isin(included_days))

    # Replace values to avoid data privacy concerns
    sub_xr['mslp'] = sub_xr['mslp'].where(False, other=random.uniform(0, 1))
    sub_xr['z500'] = sub_xr['z500'].where(False, other=random.uniform(0, 1))

    # Make data perfectly separable
    sub_xr['mslp'] = sub_xr['mslp'].where(sub_xr['labels'] == 11.0, 2)
    sub_xr['z500'] = sub_xr['z500'].where(sub_xr['labels'] == 11.0, 2)
    sub_xr['mslp'] = sub_xr['mslp'].where(sub_xr['labels'] == 17.0, 3)
    sub_xr['z500'] = sub_xr['z500'].where(sub_xr['labels'] == 17.0, 3)

    # Save as netcdf
    sub_xr.to_netcdf(path=path_unit_test_data, engine='netcdf4')
    sub_xr.close()


if __name__ == '__main__':
    generate_data_for_unit_tests()
