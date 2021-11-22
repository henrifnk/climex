from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from post_analysis.extract_time import extract_time

def plot_for_date(ds, date):
    date_formatted = datetime.strptime(date, '%Y-%m-%d')
    diff = date_formatted  - datetime(1900, 1, 1, 12, 0, 0)
    index = diff.days + 0.5
    mspl = ds.variables['mslp'][index, :, :]
    z500 = ds.variables['z500'][index, :, :]
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    label = ds.variables['labels'][index]
    # Plot
    fig = plt.figure()
    plt.figure(figsize=(14,14))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, mspl, transform = ccrs.PlateCarree())
    line_c = ax.contour(lon, lat, z500, colors=['black'], transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(mm, cax=cax)
    ax.clabel(line_c, colors=['black'], manual=False, inline=True, fmt=' {:.0f} '.format)
    if label == 11:
        label_str = 'Tief Mitteleuropa'
    elif label == 17:
        label_str = 'Trog Mitteleuropa'
    else:
        label_str = 'Residual'
    title = date + ' with label ' + label_str
    ax.set_title(title, fontsize=22)
    plt.show()

def get_dates(path_to_data='climex/data/small_trainingset/training_database_daily.nc', gwl=11):
    dataset = netcdf_dataset(path_to_data)
    time = dataset.variables['time'][:]
    time_str = extract_time(path_to_data=path_to_data, save=False)
    label = dataset.variables['labels'][:]
    index = np.where(label == gwl)
    return np.array(time_str)[index]



path_to_data = "data/training_database_daily.nc"
dataset = netcdf_dataset(path_to_data)