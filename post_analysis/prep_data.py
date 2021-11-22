import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import netCDF4
import numpy as np
from climex.models.model_conv_lstm import ResCNNEncoder, DecoderRNN
from climex.data.load_data import load_data


# os.chdir('..')
# os.listdir()


def init_resnet(model_path='post_analysis/results/ResNet_kstr4/', kernel_size=4, stride=1, padding=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = torchvision.models.resnet18()
    # Use 2-dim kernel, since our image only has 2 channel
    resnet.conv1 = nn.Conv2d(2, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    num_factors = resnet.fc.in_features
    # Set the size of each output sample to 3
    resnet.fc = nn.Linear(num_factors, 3)
    resnet.load_state_dict(torch.load(model_path + 'ResNet.pt', map_location=torch.device(device)))
    resnet.eval()
    return resnet


def init_convlstm(model_path='post_analysis/results/ConLSTM_slide_4d/', h_RNN_layers=3, h_RNN=16, h_FC_dim=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_encoder = ResCNNEncoder().to(device)
    cnn_encoder.load_state_dict(torch.load(model_path + 'CNN.pt', map_location=torch.device(device)))
    cnn_encoder.eval()
    rnn_decoder = DecoderRNN(h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim, source=-1).to(device)
    rnn_decoder.load_state_dict(torch.load(model_path + 'RNN.pt', map_location=torch.device(device)))
    rnn_decoder.eval()
    return cnn_encoder, rnn_decoder


def load_time_label(path_to_data="climex/data/entire_trainingset/training_database_3hourly.nc"):
    dataset = netCDF4.Dataset(path_to_data)
    label = torch.tensor(dataset.variables['labels'][:])
    label[label == 11] = int(1)
    label[label == 17] = int(2)
    time = torch.tensor(dataset.variables['time'][:])
    return time, label


def prep_resnet_pred(path_to_data='climex/data/entire_trainingset/training_database_3hourly.nc',
                     model_path='post_analysis/results/ResNet_kstr4/',
                     save_data_path='post_analysis/Models/resnet_data.pt',
                     kernel_size=4, stride=1, padding=3):
    resnet = init_resnet(model_path=model_path, kernel_size=kernel_size, stride=stride, padding=padding)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, val_loader = load_data(batch_size=256, n_years_val=10, n_years_test=10,
                                                      shuffle_train_data=False,
                                                      path_to_data=path_to_data)
    out = []
    output = []
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs = Variable(inputs).to(device, dtype=torch.float)
                labels = Variable(labels).to(device, dtype=torch.int64)
                outputs = resnet(inputs)
                output.append(outputs)
                out.append(outputs.detach().cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted.detach().cpu().numpy())
                true_labels.append(labels.to('cpu').numpy())
    resnet_prediction = torch.tensor(np.concatenate(pred_labels, axis=0))
    all_true_labels = torch.tensor(np.concatenate(true_labels, axis=0))
    time, label = load_time_label(path_to_data=path_to_data)
    out_all = np.vstack(out)
    length = len(time)
    resnet_data = torch.cat((time.reshape((length, 1)),
                             label.reshape((length, 1)),
                             resnet_prediction.reshape((length, 1)),
                             torch.tensor(out_all)), 1)
    torch.save(resnet_data, save_data_path)

def prep_convlstm_pred(path_to_data='climex/data/entire_trainingset/training_database_3hourly.nc',
                       model_path='climex/post_analysis/results/ConLSTM_slide/', time_depth=24,
                       save_data_path='climex/post_analysis/Models/', h_RNN_layers=3, h_RNN=16, h_FC_dim=8):
    cnn_encoder, rnn_decoder = init_convlstm(model_path=model_path, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, val_loader = load_data(batch_size=256, n_years_val=10, n_years_test=10,
                                                      key='sliding_video', time_depth=time_depth, target=-1,
                                                      shuffle_train_data=False, path_to_data=path_to_data)
    out = []
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs = Variable(inputs).to(device, dtype=torch.float)
                labels = Variable(labels).to(device, dtype=torch.int64)
                outputs_cnn = cnn_encoder(inputs)
                outputs_rnn = rnn_decoder(outputs_cnn)
                out.append(outputs_rnn.detach().cpu().numpy())
                _, predicted = torch.max(outputs_rnn.data, 1)
                pred_labels.append(predicted.detach().cpu().numpy())
                true_labels.append(labels.to('cpu').numpy())
    conv_lstm_prediction = torch.tensor(np.concatenate(pred_labels, axis=0))
    all_true_labels = torch.tensor(np.concatenate(true_labels, axis=0))
    length = len(conv_lstm_prediction)
    time, label = load_time_label(path_to_data=path_to_data)
    conv_lstm_data = torch.cat(((time[:length]).reshape((length, 1)),
                                (label[:length]).reshape((length, 1)),
                                conv_lstm_prediction.reshape((length, 1)),
                                outputs_rnn), 1)
    torch.save(conv_lstm_data, save_data_path + 'conv_lstm_data.pt')

