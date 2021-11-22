import unittest
from numpy.random import randint
from climex.models.Logger import LogModel
import logging
from testfixtures import LogCapture


class TestLogger(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.lg = LogCapture()
        cls.model_dir = "climex/tests/testdata/"
        cls.log = LogModel(model_dir=cls.model_dir, model_name="Test")
        cls.log.log_weights(use_weight=True, weight=[1, 2, 3])
        cls.log.log_epoch(metrics_="abc", stop=True)
        cls.log.log_inference(truth=randint(low=0, high=2, size=10),
                              prediction=randint(low=0, high=2, size=10))

    def test_Logger(self):
        self.lg.check_present(('climex.models.Logger', 'INFO', '===========Config Dict============'),
                              ('climex.models.Logger', 'INFO', '===========Use Weighted loss============'),
                              ('climex.models.Logger', 'INFO', '[1, 2, 3]'),
                              ('climex.models.Logger', 'INFO', '===========Start training============'),
                              ('climex.models.Logger', 'INFO', 'abc'),
                              ('climex.models.Logger', 'INFO', '===========Early stopping============'),
                              ('climex.models.Logger', 'INFO', '===========Confusion matrix============'),
                              ('climex.models.Logger', 'INFO', '===========MCC============')
                              )

    def tearDown(self):
        """
        Clean up. Deletes logging files to reduce package size.
        """
        # Remove all process connections to the log file.
        # Otherwise the file cannot be changed as used by those processes.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        LogCapture.uninstall_all()


if __name__ == '__main__':
    unittest.main()
