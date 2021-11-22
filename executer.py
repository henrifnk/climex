# -*- coding: utf-8 -*-

# TEMPORARY DEV FILE

# Run tests
# import unittest
# loader = unittest.TestLoader()
# tests = loader.discover('.')
# testRunner = unittest.runner.TextTestRunner()
# testRunner.run(tests)

from climex.models.train_resnet import train_resnet
from climex.models.train_conv_lstm import train_conv_lstm
from climex.models.train_conv_lstm_sliding import train_conv_lstm_sliding
from climex.models.train_vit import train_vit
from climex.models.train_cbam_resnet import train_cbam_resnet

train_resnet(batch_size=1, epochs=1, patience=3, lr=0.01, use_weight=True, weights=[0.5, 0.6, 0.8],
             model_dir="climex/models/results/ResNet/")
train_conv_lstm(batch_size=64, epochs=3, lr=0.1, use_weight=True, weights=[0.5, 0.6, 0.8], time_depth=2, target=2,
                model_dir="climex/models/results/LSTM/")
train_conv_lstm_sliding(batch_size=64, epochs=3, lr=0.1, use_weight=True, weights=[0.5, 0.6, 0.8], time_depth=2,
                        target=2, model_dir="climex/models/results/LSTM_sliding/", source=1)
train_vit(batch_size=64, epochs=1, patience=None, lr=0.1, use_weight=True, weights=[0.5, 0.6, 0.8],
          model_dir="climex/models/results/ViT/")
train_cbam_resnet(batch_size=1, epochs=1, patience=3, lr=0.01, use_weight=True, weights=[0.5, 0.6, 0.8],
                  model_dir="climex/models/results/CBAMResNet/")


# def main():
#     train_resnet(batch_size=64, epochs=1, patience=3, lr=0.01, use_weight=True, weights=[0.5, 0.6, 0.8])


# if __name__ == '__main__':
#     main()

# python executer.py
