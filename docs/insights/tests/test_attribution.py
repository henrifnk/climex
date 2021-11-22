import unittest
import numpy as np
from docs.insights.attribution import get_attribution
from docs.insights.input import get_input_data
from docs.insights.pretrained_model import get_model


class TestAttribution(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.model_dir = 'docs/insights/tests/models/ResNet/'
        cls.inputs, cls.labels = get_input_data(sample_size=1, path_to_data=cls.path_to_data)
        cls.model = get_model(model_type='ResNet',
                              pretrained=True,
                              model_dir=cls.model_dir,
                              path_to_data=cls.path_to_data)
        cls.attribution_pos = get_attribution(cls.model, cls.inputs, sign='positive',
                                              outlier_perc=1, target=cls.labels, noise_tunnel=False)
        cls.attribution_nt = get_attribution(cls.model, cls.inputs, sign='positive',
                                             outlier_perc=1, target=cls.labels, noise_tunnel=True)

    # Tests ---------------------------------------------------------------------------------------
    def test_get_attribution_output_format(self):
        """
        Test get_attribution function for right shape.
        """
        # attribution should consist of a array with the two dimensions (16,39)
        self.assertEqual(self.attribution_pos.shape[0], 16)
        self.assertEqual(self.attribution_pos.shape[1], 39)
        self.assertEqual(self.attribution_pos.ndim, 2)

    def test_get_attribution_sign_positive(self):
        """
        Test get_attribution function, if there are only positive values for ´positive´.
        """
        # attribution should not consist of negative values
        self.assertFalse(np.any(self.attribution_pos < 0))

    def test_get_attribution_normalized(self):
        """
        Test get_attribution function, if result are normalized.
        """
        # noise tunnel should change the attributions
        self.assertTrue(np.any(self.attribution_pos <= 1))

    def test_get_attribution_noisetunnel(self):
        """
        Test get_attribution function, if result changes with noise tunnel.
        """
        # noise tunnel should change the attributions
        self.assertTrue(np.any(self.attribution_pos != self.attribution_nt))


if __name__ == '__main__':
    unittest.main()