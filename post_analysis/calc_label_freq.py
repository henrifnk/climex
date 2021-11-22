import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def split_label(label):
    idx = np.where(np.diff(label))
    idx = [x + 1 for x in idx]
    idx = list(idx[0])
    l_list = [label[i : j] for i, j in zip([0] + idx, idx + [None])]
    return l_list

def get_label(label, day=8, classifier='Hess and Brezowsky'):
    # count the unique lables per sequence in the input set
    l_list = split_label(label)
    label_ps = []
    [label_ps.append(i[0]) for i in l_list]
    data = pd.DataFrame({'GWL': label_ps})
    seq_sum = []
    [seq_sum.append(len(i)/day) for i in l_list]
    data['days'] = np.round(seq_sum)
    data['classifier'] = classifier
    data = rename_label(data)
    return data

def rename_label(data):
    gwl = data['GWL']
    data['GWL'] = 'Residual'
    data.loc[gwl == 1, 'GWL'] = 'Tief Mitteleuropa'
    data.loc[gwl == 2, 'GWL'] = 'Trog Mitteleuropa'
    return data


def create_all_data():
    resnet_data = torch.load('post_analysis/Models/resnet_data.pt', map_location=torch.device('cpu')).detach().numpy()
    conv_lstm_data = torch.load('post_analysis/results/ConLSTM_slide_gnoise/conv_lstm_data.pt', map_location=torch.device('cpu')).detach().numpy()

    true_label = resnet_data[:, 1]
    dataset = get_label(true_label)

    resnet_label = resnet_data[:, 2]
    dataset = dataset.append(get_label(resnet_label, classifier="ResNet"))

    lstm_label = conv_lstm_data[:, 0]
    dataset = dataset.append(get_label(lstm_label, classifier="ConvLSTM"))
    return dataset



def calc_rel_ctab(data="HB", bins=[0,3,6,9,12,15]):
    resnet_data = torch.load('post_analysis/Models/resnet_data.pt', map_location=torch.device('cpu')).detach().numpy()
    conv_lstm_data = torch.load('post_analysis/Models/conv_lstm_data.pt', map_location=torch.device('cpu')).detach().numpy()

    if data == "HB":
        true_label = resnet_data[:, 1]
        dataset = get_label(true_label)

    if data == "ResNet":
        resnet_label = resnet_data[:, 2]
        dataset = get_label(resnet_label, classifier="ResNet")

    if data == "ConvLSTM":
        lstm_label = conv_lstm_data[:, 2]
        dataset = get_label(lstm_label, classifier="ConvLSTM")
    if data == "Tief":
        dataset = create_all_data()
        dataset = dataset[dataset['GWL'] != 'Residual']
        tief = dataset[dataset['GWL'] == 'Tief Mitteleuropa']
        tief['day'] = pd.cut(tief['days'], bins=bins)
        ctab = pd.crosstab(index=tief['classifier'], columns=tief['day'])
        return ctab.apply(lambda r: r / r.sum(), axis=1)

    if data == "Trog":
        dataset = create_all_data()
        dataset = dataset[dataset['GWL'] != 'Residual']
        trog = dataset[dataset['GWL'] == 'Trog Mitteleuropa']
        trog['day'] = pd.cut(trog['days'], bins=bins)
        ctab = pd.crosstab(index=trog['classifier'], columns=trog['day'])
        return ctab.apply(lambda r: r / r.sum(), axis=1)

    dataset['day'] = pd.cut(dataset['days'], bins=bins)
    ctab = pd.crosstab(index=dataset['GWL'], columns=dataset['day'])
    return ctab.apply(lambda r: r/r.sum(), axis=1)



def plot_label_frequencies(save=True, omit_residual=True, bins=np.arange(16)):
    dataset = create_all_data()
    dataset = dataset.reset_index()
    if omit_residual:
        dataset = dataset[dataset['GWL'] != 'Residual']
    g = sns.FacetGrid(dataset, col="GWL",  row="classifier", sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="days", stat='density', bins=bins)
    if save:
        plt.savefig('post_analysis/Models/distrib.png', dpi=400)
    plt.show()