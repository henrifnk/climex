import os
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
from climex.models.model_vit import ViT
from climex.models.Logger import LogModel
import torch
import torch.nn as nn
import numpy as np
from climex.data.load_data import load_data
from climex.models.utils import EarlyStopping, setup_weights, check_common_args


def train_vit(batch_size=64, epochs=10, patience=None, lr=5e-5, use_weight=True,
              weights=None, n_years_val=10, n_years_test=10, splitting_method='sequential', season=None,
              path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc',
              model_dir="climex/models/results/"):
    """Train ViT model.

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
        season (str, optional): setting it to None means that all images will be loaded.
            Alternatively, it can be set to 'winter' or 'summer' to only load images
            from winter or summer months respectively.
        path_to_data (str): Path to the nc file with the data to be loaded.
        model_dir (str): Path to store the model in.

    Return:
        ViT.pt and logging.log files, saved in the model_dir.
    """
    if patience is None:
        patience = epochs
    check_common_args(batch_size, epochs, patience, lr, use_weight, weights, n_years_val,
                      n_years_test, splitting_method, season, path_to_data, model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # Set Up logging and add locals from train_resnet env
    logger = LogModel(model_dir=model_dir, model_name="ViT")
    logger.logger.info(locals())

    # Load, split and preprocess data (images)
    train_loader, test_loader, val_loader = load_data(
        batch_size=batch_size,
        n_years_val=n_years_val,
        n_years_test=n_years_test,
        splitting_method=splitting_method,
        season=season,
        path_to_data=path_to_data
    )

    # Obtain manual or computed weighted loss, if required
    if use_weight:
        weights = setup_weights(train_loader=train_loader, weights=weights)
    logger.log_weights(use_weight=use_weight, weight=weights)

    model = ViT().to(device)

    # Loss and optimizer
    criterion = nn.NLLLoss(weight=weights)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    early_stopping = EarlyStopping(model_dir=model_dir, patience=patience,
                                   verbose=True, name='ViT.pt')

    loss_hist = {}
    loss_hist["train accuracy"] = []
    loss_hist["train loss"] = []
    loss_hist["valid accuracy"] = []
    loss_hist["valid loss"] = []

    # Train the model
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for batch_idx, (img, labels) in enumerate(train_loader):
            img = img.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.int64)

            preds = model(img)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        epoch_valid_loss = 0
        y_true_valid = []
        y_pred_valid = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (img, labels) in enumerate(val_loader):
                img = img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.int64)
                preds = model(img)
                loss = criterion(preds, labels)

                y_pred_valid.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_valid.extend(labels.detach().tolist())

                epoch_valid_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)
        loss_hist["valid loss"].append(epoch_valid_loss)
        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x == y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total
        valid_correct = len([True for x, y in zip(y_pred_valid, y_true_valid) if x == y])
        valid_total = len(y_pred_valid)
        valid_accuracy = valid_correct * 100 / valid_total

        loss_hist["train accuracy"].append(accuracy)
        loss_hist["valid accuracy"].append(valid_accuracy)

        mi_f1_score = metrics.f1_score(y_true_train, y_pred_train, average='micro')
        ma_f1_score = metrics.f1_score(y_true_train, y_pred_train, average='macro')
        w_f1_score = metrics.f1_score(y_true_train, y_pred_train, average='weighted')
        mcc = metrics.matthews_corrcoef(y_true_train, y_pred_train)

        metric_parse = "epoch:%d, micro_f1:%f, macro_f1:%f, weighted_f1:%f, training accuracy: " \
                       "%f,training loss:%f,  validation loss:%f, mcc:%f" % \
                       (epoch, mi_f1_score, ma_f1_score, w_f1_score, accuracy, epoch_train_loss,
                        epoch_valid_loss, mcc)
        early_stopping(val_loss=epoch_valid_loss, model=model)
        logger.log_epoch(metrics_=metric_parse, stop=early_stopping.early_stop)
        if early_stopping.early_stop:
            break

    true_labels = []
    pred_labels = []
    out = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = Variable(images).to(device, dtype=torch.float)
            labels = Variable(labels).to(device, dtype=torch.int64)
            outputs = model(images)
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
