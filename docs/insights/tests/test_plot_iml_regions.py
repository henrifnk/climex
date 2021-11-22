import os
import unittest

from docs.insights.input import get_input_data
from docs.insights.plot_iml_regions import plot_iml_regions
from docs.insights.pretrained_model import get_model


class TestPlot(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.inputs, _ = get_input_data(sample_size=1, path_to_data=cls.path_to_data)
        cls.fig_name1 = 'test_plot'
        cls.fig_path = 'docs/insights/tests/plots/'
        cls.model_dir = 'docs/insights/tests/models/ResNet/'
        cls.model = get_model(model_type='ResNet',
                              pretrained=True,
                              model_dir=cls.model_dir,
                              path_to_data=cls.path_to_data)

        if not os.path.exists(cls.fig_path):
            os.makedirs(cls.fig_path)

    # Tests ---------------------------------------------------------------------------------------
    def test_plot(self):
        """
        Check, if file of the plot exists in the end.
        """
        plot_iml_regions(self.inputs,
                         self.model,
                         sign='positive',
                         outlier_perc=2,
                         target=-1,
                         noise_tunnel=False,
                         fig_name=self.fig_name1,
                         fig_path=self.fig_path,
                         path_to_data=self.path_to_data)

        self.assertTrue(os.path.isfile((self.fig_path + self.fig_name1 + '.png')))

    def tearDown(self):
        """
        Clean up. Deletes plot file to reduce package size.
        """
        for fi in os.listdir(self.fig_path):
            os.remove(os.path.join(self.fig_path, fi))


if __name__ == '__main__':
    unittest.main()