import unittest
from climex.models.train_conv_lstm_sliding import train_conv_lstm_sliding
import os
import warnings


class TestTrainConvLSTMSliding(unittest.TestCase):

    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        cls.path_to_data = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.model_dir = 'climex/tests/testmodels_results/ConvLSTMSliding/'
        if not os.path.exists(cls.model_dir):
            os.makedirs(cls.model_dir)

    def test_argument_checks(self):
        # Test for checks for `batch_size`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(batch_size=-1)
        self.assertEqual(cm.exception.args[0], '`batch_size` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(batch_size="1")
        self.assertEqual(cm.exception.args[0], '`batch_size` must be an integer.')

        # Test for `epochs`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(epochs=-1)
        self.assertEqual(cm.exception.args[0], '`epochs` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(epochs="1")
        self.assertEqual(cm.exception.args[0], '`epochs` must be an integer.')

        # Test for `patience`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(patience=-1)
        self.assertEqual(cm.exception.args[0], '`patience` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(patience="1")
        self.assertEqual(cm.exception.args[0], '`patience` must be an integer.')

        # Test for `lr`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(lr=-0.1)
        self.assertEqual(cm.exception.args[0], '`lr` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(lr="1")
        self.assertEqual(cm.exception.args[0], '`lr` must be a float.')

        # Tests for `use_weight`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(use_weight='not_a_bool')
        self.assertEqual(cm.exception.args[0], "`use_weight` must be a bool.")
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(use_weight=1)
        self.assertEqual(cm.exception.args[0], "`use_weight` must be a bool.")

        # Tests for `weights`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(weights=1)
        self.assertEqual(cm.exception.args[0], "`weights` must be a float list of len 3 or None.")
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(weights=[1, 1, '1'])
        self.assertEqual(cm.exception.args[0], "`weights` must be a float list of len 3 or None.")

        # Tests for `n_years_val` and `n_years_test`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(n_years_val=-1)
        self.assertEqual(cm.exception.args[0], '`n_years_val` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(n_years_val="1")
        self.assertEqual(cm.exception.args[0], '`n_years_val` must be an integer.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(n_years_test=-1)
        self.assertEqual(cm.exception.args[0], '`n_years_test` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(n_years_test="1")
        self.assertEqual(cm.exception.args[0], '`n_years_test` must be an integer.')

        # Tests for checks for `splitting_method`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(splitting_method='unknown_method')
        self.assertEqual(cm.exception.args[0], "`splitting_method` must be 'sequential' or 'random'.")

        # Tests for checks for `season`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(season='unknown_season')
        self.assertEqual(cm.exception.args[0], "`season` must be None, 'winter' or 'summer'.")

        # Test for `source`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(source=-1)
        self.assertEqual(cm.exception.args[0], '`source` must be greater than 0.')
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(source="1")
        self.assertEqual(cm.exception.args[0], '`source` must be an integer.')

        # Tests for checks for `path_to_data`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(path_to_data=list())
        self.assertEqual(cm.exception.args[0], "`path_to_data` must be a string.")
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(path_to_data='test_file.csv')
        self.assertEqual(cm.exception.args[0], "Can only load netcdf files with file extension '.nc'.")
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(path_to_data='unknown_path.nc')
        self.assertEqual(cm.exception.args[0], "`path_to_data` does not exists.")

        # Tests for `model_dir`
        with self.assertRaises(SystemExit) as cm:
            train_conv_lstm_sliding(model_dir=list())
        self.assertEqual(cm.exception.args[0], "`model_dir` must be a string.")

    def test_model_comp_weigths(self):
        """
        Test model computed weights in loss function.
        """
        if os.getenv("CI") is not None:
            self.skipTest("Skipping testing in CI environment")
        try:
            # Hide warning that measures are ill-defined (only due to our very limited unit test data)
            warnings.filterwarnings('ignore')
            train_conv_lstm_sliding(batch_size=64, epochs=1, patience=7, use_weight=True, weights=None, source=1,
                                    model_dir=self.model_dir)
        except SystemExit:
            self.fail("model_conv_lstm_sliding() raised SystemExit unexpectedly!")

        self.assertTrue(os.path.isfile(self.model_dir + 'CNN.pt'))
        self.assertTrue(os.path.isfile(self.model_dir + 'RNN.pt'))
        self.assertTrue(os.path.isfile(self.model_dir + 'logging.log'))

    def tearDown(self):
        """
        Clean up. Deletes model and loggin files to reduce package size.
        """
        for fi in os.listdir(self.model_dir):
            os.remove(os.path.join(self.model_dir, fi))


if __name__ == '__main__':
    unittest.main()
