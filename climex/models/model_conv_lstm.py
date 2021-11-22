import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResCNNEncoder(nn.Module):
    """ResNet CNN Encoder class

    The goal of RNN models is to extract the temporal correlation between the images by keeping a memory of past
    images. The images of a video are fed to a CNN model to extract high-level features. The features are then fed to
    an RNN layer and the output of the RNN layer is connected to a fully connected layer to get the classification
    output. We will use ResNet18 as the base CNN model.

    Attributes:
        fc_hidden1 (int): size of the first fully connected hidden layer
        fc_hidden2 (int): size of the second fully connected hidden layer
        drop_p (float): drop out
        resnet (model): pytorch resnet model that is modified to a encoder by dropping the last layer.
        fc1 (torch.nn): fully connected layer
        fc2 (torch.nn): fully connected layer
        bn1 (torch.nn): batch normalization
        bn2 (torch.nn): batch normalization
    """

    def __init__(self, fc_hidden1=512, fc_hidden2=256, drop_p=0.1, CNN_embed_dim=128):
        """Load the ResNet-18 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=1, padding=3, bias=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    """ResNet Decoder class

    Attributes:
        RNN_input_size (int): RNN input size.
            Must be equal to CNN_embed_dim
        h_RNN_layers (int):  # RNN hidden layers
        h_RNN (int): # RNN hidden nodes
        drop_p (float): drop out  rate
        source (any): if source is not None, expect LSTM to be sliding, else classic LSTM.
            Default is None
        num_classes: number of labels to classify
    """
    def __init__(self, CNN_embed_dim=128, h_RNN_layers=3, h_RNN=128, h_FC_dim=64,
                 drop_p=0.1, num_classes=3, source=None):
        """Initializes the Decoder LSTM.
        """
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.source = source

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        # h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size)
        # None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size)

        # FC layers
        if self.source is None:
            x = self.fc1(RNN_out)
        else:
            x = self.fc1(RNN_out[:, self.source, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x
