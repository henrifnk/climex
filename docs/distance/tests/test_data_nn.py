import unittest
from climex.data.load_data import load_data
from docs.distance.data_nn import data_nn


class TestDataNN(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.train_loader, cls.test_loader, cls.val_loader = load_data(n_years_val=10, n_years_test=10,
                                                                      splitting_method='sequential',
                                                                      season=None, path_to_data=cls.path_to_data,
                                                                      shuffle_train_data=False)
        cls.model_path = "files/model-vit-20.pt"
        cls.output_label, cls.output_mslp = data_nn(n_years_val=10, n_years_test=10, splitting_method='sequential',
                                                    season=None, path_to_data=cls.path_to_data,
                                                    model_path=cls.model_path)

    def test_output(self):
        """
        Test whether the size of the output and output label is identical as expected
        """
        self.assertEqual(len(self.output_label), len(self.train_loader))
        self.assertEqual(len(self.output_mslp), len(self.output_label))

        input_label = []
        for _, label in self.train_loader:
            if label.numpy()[0] == 0:
                input_label.append(0)
            if label.numpy()[0] == 1:
                input_label.append(11)
            if label.numpy()[0] == 2:
                input_label.append(17)

        self.assertEqual(self.output_label, input_label)


if __name__ == '__main__':
    unittest.main()
