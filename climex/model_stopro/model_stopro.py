import torch
from climex.models.train_resnet import train_resnet
from climex.data.load_data import load_data
from post_analysis.prep_data import init_resnet

def get_encoding_dim(path_to_model='post_analysis/results/ResNet_kstr4/', kernel_size=4, stride=1, padding=3,
                     path_to_data='climex/data/entire_trainingset/training_database_3hourly.nc'):
    # load or train the model
    if path_to_model == None:
        resnet = train_resnet(path_to_data=path_to_data, kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        resnet = init_resnet(model_path=path_to_model, kernel_size=kernel_size, stride=stride, padding=padding)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, val_loader = load_data(batch_size=256, n_years_val=10, n_years_test=10,
                                                      shuffle_train_data=False, path_to_data=path_to_data)

    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
