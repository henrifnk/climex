import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as clr
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
import matplotlib

def plot_year(year, month=[0], save=False):
    resnet_data = torch.load('post_analysis/Models/resnet_data.pt', map_location=torch.device('cpu'))
    conv_lstm_data = torch.load('post_analysis/results/ConLSTM_slide_gnoise/conv_lstm_data.pt', map_location=torch.device('cpu'))
    softmax = torch.nn.Softmax(dim=1)
    resnet_data[:, [3, 4, 5]] = softmax(torch.tensor(resnet_data[:, [3, 4, 5]]))
    conv_lstm_data[:, [1, 2, 3]] = softmax(conv_lstm_data[:, [1, 2, 3]])
    resnet_data = np.array(resnet_data)
    conv_lstm_data = np.array(conv_lstm_data)
    nn = np.zeros((23, 4))
    conv_lstm_data = np.concatenate((nn, conv_lstm_data[range(265865), :],
                                    nn, conv_lstm_data[range(265865, 295066), :],
                                    nn, conv_lstm_data[range(295066, 324259), :]), axis=0)
    time = np.load('post_analysis/Models/time.npy', allow_pickle=True)
    idx = []
    for i in range(len(time)):
        if (time[i].year == year) and ((month == [0]) or (time[i].month in month)):
            idx.append(i)
    cmap_blue = clr.LinearSegmentedColormap.from_list('custom blue', ['white', 'blue'], N=256)
    cmap_green = clr.LinearSegmentedColormap.from_list('custom green', ['white', 'green'], N=256)
    time = time[idx]
    resnet_data = resnet_data[idx]
    conv_lstm_data = conv_lstm_data[idx]
    colors = {0: 'red', 1: 'green', 2: 'blue'}
    fig, ax = plt.subplots(figsize=(8, 2))
    fig = matplotlib.pyplot.gcf()
    ax.scatter(time, ['ResNet TM'] * len(time), c=pd.Series(resnet_data[:, 4]), cmap=cmap_green)
    ax.scatter(time, ['ResNet TRM'] * len(time), c=pd.Series(resnet_data[:, 5]), cmap=cmap_blue)
    ax.scatter(time, ['ResNet label'] * len(time), color=pd.Series(resnet_data[:, 2]).map(colors))
    ax.scatter(time, ['true label'] * len(time), color=pd.Series(resnet_data[:, 1]).map(colors))
    ax.scatter(time, ['Conv LSTM label'] * len(time), color=pd.Series(conv_lstm_data[:, 0]).map(colors))
    ax.scatter(time, ['Conv LSTM TRM'] * len(time), c=pd.Series(conv_lstm_data[:, 3]), cmap=cmap_blue)
    ax.scatter(time, ['Conv LSTM TM'] * len(time), c=pd.Series(conv_lstm_data[:, 2]), cmap=cmap_green)
    params = {'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16}
    pylab.rcParams.update(params)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.xlabel('date')
    plt.ylabel('predicted value')
    red_patch = mpatches.Patch(color='red', label='Residual')
    blue_patch = mpatches.Patch(color='blue', label='Trog Mitteleuropa')
    green_patch = mpatches.Patch(color='green', label='Tief Mitteleuropa')
    plt.legend(handles=[red_patch, blue_patch, green_patch], fontsize=18, loc="upper right")
    fig.suptitle('Label values ' + str(month) + ' ' + str(year), fontsize=22)
    plt.gcf().autofmt_xdate()
    if save:
        fig.savefig('post_analysis/Models/labels' + '_' + str(year) + '_' + str(month) + '.png', dpi=400)
    plt.show()
plot_year(1977, [7,8], save=True)