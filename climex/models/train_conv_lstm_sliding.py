# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
import sklearn
from climex.models.Logger import LogModel
from climex.models.model_conv_lstm import ResCNNEncoder, DecoderRNN
from climex.data.load_data import load_data
from climex.models.utils import EarlyStopping, setup_weights, check_common_args, check_int_gr_0


def train_conv_lstm_sliding(batch_size=256, epochs=100, patience=7, lr=0.01, use_weight=True, weights=None,
                            n_years_val=10, n_years_test=10, splitting_method='sequential', time_depth=8, target=8,
                            season=None, source=1, save_test=None,
                            path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc',
                            model_dir="climex/models/results/"):
    """Train Convolutional LSTM model.

    Args:
        batch_size (int): Define a batch size, defaults to 265.
        epochs (int): Maximal amount of epochs to train the model, defaults to 100.
        patience (int): Patience for early stopping, defaults to 7
        lr (float): Initial learning rate, defaults to 0.01
        use_weight (bool): Use weighted loss? Default is True
        weights (list of 3 floats or None): Only needs to be set if use_weight=True.
            List with 3 floats containing the manual weights for the loss function.
            If None then weights are computed depending on class frequency.
        n_years_val (int): the number of years to include in the validation set.
        n_years_test (int): the number of years to include in the test set.
        splitting_method (str): the splitting method executed. Must be 'sequential' or 'random'.
        time_depth (int): int of length 1 indicating video length.
        target (int): Indicating which picture of the video serves for the
            prediction label.
        season (str, optional): setting it to None means that all images will be loaded.
            Alternatively, it can be set to 'winter' or 'summer' to only load images
            from winter or summer months respectively.
        source: which hidden state should be used
        path_to_data (str): Path to the nc file with the data to be loaded.
        model_dir (str): Path to store the model in.
        save_test (str): location where to store test set
            Set to None where no saving is nessecary.

    Return:
        CNN.pt encoder, RNN.pt decoder and logging.log files, saved in the model_dir.
    """

    check_args_train_conv_lstm_sliding(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                                       splitting_method, season, source, path_to_data, model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    logger = LogModel(model_dir=model_dir, model_name="ConvLSTM_sliding")
    logger.logger.info(locals())

    # Load, split and preprocess data (images)
    train_loader, test_loader, val_loader = load_data(
        key='sliding_video',
        time_depth=time_depth,
        target=target,
        batch_size=batch_size,
        n_years_val=n_years_val,
        n_years_test=n_years_test,
        splitting_method=splitting_method,
        season=season,
        path_to_data=path_to_data,
        save_test=save_test
    )

    # Obtain manual or computed weighted loss, if required
    if use_weight:
        weights = setup_weights(train_loader, weights)
    logger.log_weights(use_weight=use_weight, weight=weights)

    cnn_encoder = ResCNNEncoder().to(device)
    rnn_decoder = DecoderRNN(h_RNN_layers=3, h_RNN=16, h_FC_dim=8, source=source).to(device)

    crnn_params = (list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) +
                   list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) +
                   list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters()))

    optimizer = torch.optim.Adam(crnn_params, lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    early_stopping = EarlyStopping(patience=patience, verbose=True, model_dir=model_dir)

    # Store our loss and accuracy for plotting
    valid_losses = []
    train_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    true_labels = []
    pred_labels = []
    out = []
    correct = 0
    total = 0
    for epoch in tqdm(range(epochs)):
        cnn_encoder.train()
        rnn_decoder.train()

        for i, data in enumerate(train_loader):
            length = len(train_loader)
            inputs, labels = data
            inputs = Variable(inputs).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            optimizer.zero_grad()

            # forward + backward
            output = rnn_decoder(cnn_encoder(inputs))  # output has dim = (batch, num_classes)
            loss = criterion(output, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)

        # Compute validation loss
        cnn_encoder.eval()
        rnn_decoder.eval()
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = Variable(inputs).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            output = rnn_decoder(cnn_encoder(inputs))

            loss = criterion(output, labels)
            valid_losses.append(loss.item())

            out.append(output.detach().cpu().numpy())
            _, predicted = torch.max(output.data, 1)
            pred_labels.append(predicted.detach().cpu().numpy())
            true_labels.append(labels.to('cpu').numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        prediction = np.concatenate(pred_labels, axis=0)
        truth = np.concatenate(true_labels, )
        mi_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='micro')
        ma_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='macro')
        w_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='weighted')
        test_accuracy = (sum(np.array(prediction) == np.array(truth)) / float(len(truth)))
        mcc = metrics.matthews_corrcoef(truth, np.array(prediction))

        # Early stopping if required
        early_stopping(valid_loss, cnn_encoder, rnn_decoder)

        metric_parse = "epoch:%d, train_loss:%f, valid_loss:%f, micro_f1:%f, macro_f1:%f, weighted_f1:%f, accuracy: " \
                       "%f, mcc:%f" % \
                       (epoch, train_loss, valid_loss, mi_f1_score, ma_f1_score, w_f1_score, test_accuracy, mcc)
        logger.log_epoch(metrics_=metric_parse, stop=early_stopping.early_stop)

        if early_stopping.early_stop:
            break
        valid_losses = []
        train_losses = []
        true_labels = []
        pred_labels = []
        out = []
        correct = 0
        total = 0

    true_labels = []
    pred_labels = []
    out = []
    correct = 0
    total = 0
    cnn_encoder.eval()
    rnn_decoder.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = Variable(images).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            output = rnn_decoder(cnn_encoder(images))
            out.append(output.detach().cpu().numpy())
            _, predicted = torch.max(output.data, 1)
            pred_labels.append(predicted.detach().cpu().numpy())
            true_labels.append(labels.to('cpu').numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum()

    prediction = np.concatenate(pred_labels, axis=0)
    truth = np.concatenate(true_labels, )
    # Compute and store accuarcy measures
    logger.log_inference(truth=truth, prediction=prediction)

    # Remove all process connections to the log file.
    # Otherwise the file cannot be changed as used by those processes.
    logger.remove_handler()


def check_args_train_conv_lstm_sliding(batch_size, epochs, patience, lr, use_weight, weights, n_years_val,
                                       n_years_test, splitting_method, season, source, path_to_data, model_dir):
    """Argument checks for `train_conv_lstm_sliding` function.

        Arguments
            See `?train_conv_lstm_sliding`.
    """

    # Checks for common args (that appear in all models)
    check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                      splitting_method, season, path_to_data, model_dir)
