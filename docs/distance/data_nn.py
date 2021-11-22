import torch
from climex.data.load_data import load_data


def data_nn(n_years_val=10, n_years_test=10, splitting_method='sequential', season=None,
            path_to_data='climex/tests/testdata/training_database_daily_unit_tests.nc',
            model_path="files/model-vit-20.pt"):
    """

    :param n_years_val: number of years for the validation set.
    :param n_years_test: number of years for the validation set.
    :param splitting_method: 'sequential' or 'random'.
    :param season: None or 'winter' or 'summer' from whole data or winter or summer months respectively.
    :param path_to_data: path to the nc file.
    :param model_path: path to the pretrained model file.
    :return: label list and feature map data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, val_loader = load_data(
        batch_size=1,
        n_years_val=n_years_val,
        n_years_test=n_years_test,
        splitting_method=splitting_method,
        season=season,
        path_to_data=path_to_data,
        shuffle_train_data=False,
    )

    model = torch.load(model_path, map_location=device)

    # Test the model
    with torch.no_grad():
        model.eval()

        mslp = []
        mslp_labels = []

        for batch_idx, (img, labels) in enumerate(train_loader):
            img = img.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.int64)

            model(img)

            feature = model.featuremap.transpose(1, 0).cpu().numpy().reshape(-1)
            mslp.append(feature)

            if labels.numpy()[0] == 0:
                mslp_labels.append(0)
            if labels.numpy()[0] == 1:
                mslp_labels.append(11)
            if labels.numpy()[0] == 2:
                mslp_labels.append(17)

    return mslp_labels, mslp
