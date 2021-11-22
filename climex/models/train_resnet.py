# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
import sklearn
from sklearn import metrics
import time as time
from climex.models.Logger import LogModel
from climex.data.load_data import load_data
from climex.models.utils import adjust_learning_rate, EarlyStopping, setup_weights, check_common_args, check_int_gr_0


# TODO: Improve docstring
def train_resnet(batch_size=256, epochs=100, patience=7, lr=0.01, kernel_size=7, stride=2, padding=3, use_weight=True,
                 weights=None, n_years_val=10, n_years_test=10, splitting_method='sequential', season=None, save_test=None,
                 path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc',
                 model_dir="climex/models/results/"):
    """Trains and saves a ResNet18 model.

       Args:
          batch_size (int): Defines the number of samples that will be propagated through the network in one epoch.
          epochs (int): Number of epochs.
          patience (int): How many epochs to wait with early stopping after validation loss increases.
          lr (float): Learning rate.
          kernel_size (int): The size of the kernel.
          stride (int): The size of stride.
          padding (int): The size of padding.
          use_weight (bool): True for weighted loss function.
          weights (list of 3 floats or None): Only needs to be set if use_weight=True.
            List with 3 floats containing the manual weights for the loss function.
            If None then weights are computed depending on class frequency.
          n_years_val (int): The number of years to include in the validation set.
          n_years_test (int): The number of years to include in the test set.
          splitting_method (char): The splitting method executed. Must be 'sequential' or 'random'.
          season (char or None): Setting it to None means that all images will be loaded.
                                 Alternatively, it can be set to 'winter' or 'summer' to only load images
                                 from winter or summer months respectively.
          save_test (str): location where to store test set
                                 Set to None where no saving is nessecary.
         path_to_data: Path to the nc file with the entire dataset.
         model_dir: Directory in which the trained model and the log file are saved.
    """

    check_args_train_resnet(batch_size, epochs, patience, lr, kernel_size, stride, padding, use_weight, weights,
                            n_years_val, n_years_test, splitting_method, season, path_to_data, model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If not exists then create folder for model results
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Set Up logging and add locals from train_resnet env
    logger = LogModel(model_dir=model_dir, model_name="ResNet")
    logger.logger.info(locals())

    # Load, split and preprocess data (images)
    train_loader, test_loader, val_loader = load_data(
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
        weights = setup_weights(train_loader=train_loader, weights=weights)
    logger.log_weights(use_weight=use_weight, weight=weights)

    # Set model architecture
    # Use ResNet18
    model = torchvision.models.resnet18()
    # Use 2-dim kernel, since our image only has 2 channel
    model.conv1 = nn.Conv2d(2, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    num_ftrs = model.fc.in_features
    # Set the size of each output sample to 3
    model.fc = nn.Linear(num_ftrs, 3)

    # Set up model specifications
    net = model.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    early_stopping = EarlyStopping(model_dir=model_dir, patience=patience, verbose=True)

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

    # Train the model
    for epoch in tqdm(range(epochs)):
        adjust_learning_rate(lr, optimizer, epoch)
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)

        # Compute validation loss
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs = Variable(inputs).to(device, dtype=torch.float)
                labels = Variable(labels).to(device, dtype=torch.int64)
                outputs = net(inputs)
                out.append(outputs.detach().cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted.detach().cpu().numpy())
                true_labels.append(labels.to('cpu').numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum()
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

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
        early_stopping(val_loss=valid_loss, model=model)
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

    # Test the model
    tick = time.time()
    true_labels = []
    pred_labels = []
    out = []
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = Variable(images).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            outputs = net(images)
            out.append(outputs.detach().cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            pred_labels.append(predicted.detach().cpu().numpy())
            true_labels.append(labels.to('cpu').numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum()

    prediction = np.concatenate(pred_labels, axis=0)
    truth = np.concatenate(true_labels, )

    # Compute and store accuracy measures
    logger.log_inference(truth=truth, prediction=prediction)

    # Remove all process connections to the log file.
    # Otherwise the file cannot be changed as used by those processes.
    logger.remove_handler()


def check_args_train_resnet(batch_size, epochs, patience, lr, kernel_size, stride, padding, use_weight, weights,
                            n_years_val, n_years_test, splitting_method, season, path_to_data, model_dir):
    """Argument checks for `train_resnet` function.
    """

    # Checks for common args (that appear in all models)
    check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val, n_years_test,
                      splitting_method, season, path_to_data, model_dir)

    # Checks for kernel_size
    check_int_gr_0('kernel_size', kernel_size)

    # Checks for stride
    check_int_gr_0('stride', stride)

    # Checks for padding
    check_int_gr_0('padding', padding)
