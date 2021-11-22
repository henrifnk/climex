import os
import unittest
import warnings
import torchvision.models
from docs.insights.pretrained_model import get_model


class TestPretrained(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.model_dir = 'docs/insights/tests/models/ResNet/'
        if not os.path.exists(cls.model_dir):
            os.makedirs(cls.model_dir)

    # Tests ---------------------------------------------------------------------------------------
    def test_get_trained_model(self):
        """
        Tests, if the model is trained, if pretrained false
        """

        try:
            # Hide warning that measures are ill-defined (only due to our very limited unit test data)
            warnings.filterwarnings('ignore')
            model = get_model(model_type='ResNet',
                              pretrained=False,
                              model_dir=self.model_dir,
                              path_to_data=self.path_to_data)
        except SystemExit:
            self.fail("train_resnet() raised SystemExit unexpectedly!")

        self.assertTrue(os.path.isfile(self.model_dir + 'ResNet.pt'))
        self.assertTrue(os.path.isfile(self.model_dir + 'logging.log'))
        self.assertIsInstance(model, torchvision.models.resnet.ResNet)

    def test_get_trained_model_pretrained(self):
        """
        Tests, if the model is trained, if pretrained true
        """
        try:
            # Hide warning that measures are ill-defined (only due to our very limited unit test data)
            warnings.filterwarnings('ignore')
            model = get_model(model_type='ResNet',
                              pretrained=True,
                              model_dir=self.model_dir,
                              path_to_data=self.path_to_data)
        except SystemExit:
            self.fail("train_resnet() raised SystemExit unexpectedly!")

        self.assertIsInstance(model, torchvision.models.resnet.ResNet)


if __name__ == '__main__':
    unittest.main()