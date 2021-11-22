import numpy as np
import logging
from time import time
from sklearn import metrics


class LogModel(object):
    """ Class that is used during model training that saves modelling in a unified text.

    Attributes:
        model_dir (str): Hints the logger where to save the logging file
        model_name (str): Define the models name
        logger (object): logger from logging.getLogger
        time (float): saves the time to calculate train, test and validation time
    """
    def __init__(self, model_dir, model_name):
        """Initializes logging by setting up a logger for the current model
        """
        logging.basicConfig(filename=model_dir + 'logging.log', level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info('=================================' + model_name + '==================================')
        self.logger.info('===========Config Dict============')
        self.time = time()

    def log_weights(self, use_weight, weight):
        """Saves Weights and initializes header for training

        Args:
            use_weight (bool): True for weighted loss function.
            weight (FloatTensor or None): FloatTensor with 3 floats containing the manual weights for the loss
                function. If None 'use unweighted loss'
        """
        if use_weight:
            self.logger.info('===========Use Weighted loss============')
            self.logger.info(weight)
        else:
            self.logger.info('===========Use Unweighted loss============')
        self.logger.info('===========Start training============')

    def save_time(self, stage='Training'):
        """Saves the time a given modelling stage takes, logs it and re-initializes the timer.

            Args:
                stage (string): Modelling stage for logged time.
        """
        self.logger.info("===========Total " + stage + " Time============")
        self.logger.info(time() - self.time)
        self.time = time()

    def log_epoch(self, metrics_, stop):
        """Log metrics of any epoch and early stopping

        Args:
            metrics_ (string): A string, containing information of epochs metrics.
            stop (bool): Is early stopping triggered in this epoch?
        """
        self.logger.info(metrics_)
        if stop:
            self.logger.info("===========Early stopping============")
            self.save_time(stage='Training')

    def log_inference(self, truth, prediction):
        """Log metrics of any epoch and early stopping.

        Args:
            truth (array): An array, containing true labels of the test set
            prediction (array): An array, containing predicted labels of the test set
        """
        self.logger.info("=================================Inference Results==================================")
        self.save_time(stage='Inference')
        self.logger.info("===========Confusion matrix============")
        self.logger.info(metrics.confusion_matrix(truth, np.array(prediction)))
        self.logger.info("===========Classification report============")
        self.logger.info(metrics.classification_report(truth, np.array(prediction)))
        self.logger.info("===========MCC============")
        self.logger.info(metrics.matthews_corrcoef(truth, np.array(prediction)))

    def remove_handler(self):
        """Remove all handlers
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
