import netCDF4
import numpy as np
def extract_time(path_to_data = "climex/data/entire_trainingset/training_database_3hourly.nc",
                 save_data_path= 'post_analysis/Models/time.npy', save=True):
    dataset = netCDF4.Dataset(path_to_data)
    time = netCDF4.num2date(dataset.variables['time'],dataset.variables['time'].units).data
    time_r = []
    for i in range(len(time)):
        time_r.append(time[i]._to_real_datetime())
    if save:
        np.save(save_data_path, time_r)
    else:
        return time_r