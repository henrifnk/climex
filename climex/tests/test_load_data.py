import unittest
from torch.utils.data import DataLoader
from climex.data.load_data import load_data


class TestLoadData(unittest.TestCase):
    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        path = 'climex/tests/testdata/training_database_daily_unit_tests.nc'
        cls.time_depth = 3
        # test default img, video and sliding video
        cls.img = load_data(path_to_data=path)
        cls.vid = load_data(key='video', time_depth=cls.time_depth,
                            target=-1, path_to_data=path)
        cls.sld_vid = load_data(key='sliding_video', time_depth=cls.time_depth,
                                target=-1, path_to_data=path)

    def test_argument_checks(self):
        """Check argument validity.
        """
        # Test for checks for `key`
        with self.assertRaises(SystemExit) as cm:
            load_data(key='unknown_key')
        self.assertEqual(cm.exception.args[0], "`key` must be 'image', 'video' or 'sliding_video'.")

        # Test for checks for `time_depth` and `target`
        with self.assertRaises(SystemExit) as cm:
            load_data(key='video')
        self.assertEqual(cm.exception.args[0], "`time_depth` must be an integer for `key`s other than 'image'.")

        with self.assertRaises(SystemExit) as cm:
            load_data(key='video', time_depth=-1)
        self.assertEqual(cm.exception.args[0], "`time_depth` must be greater than 0.")

        with self.assertRaises(SystemExit) as cm:
            load_data(key='sliding_video', time_depth=1, target=-2)
        self.assertEqual(cm.exception.args[0], "`target` cannot be greater than `time_depth`.")

        # Test for checks for `batch_size`
        with self.assertRaises(SystemExit) as cm:
            load_data(batch_size=-1)
        self.assertEqual(cm.exception.args[0], '`batch_size` must be greater than 0.')

        with self.assertRaises(SystemExit) as cm:
            load_data(batch_size="1")
        self.assertEqual(cm.exception.args[0], '`batch_size` must be an integer.')

        # Test for checks for `n_years_val` and `n_years_test`
        with self.assertRaises(SystemExit) as cm:
            load_data(n_years_val=-1)
        self.assertEqual(cm.exception.args[0], '`n_years_val` and `n_years_test` must be greater than 0.')

        with self.assertRaises(SystemExit) as cm:
            load_data(n_years_test=-1)
        self.assertEqual(cm.exception.args[0], '`n_years_val` and `n_years_test` must be greater than 0.')

        with self.assertRaises(SystemExit) as cm:
            load_data(n_years_val="1")
        self.assertEqual(cm.exception.args[0], '`n_years_val` and `n_years_test` must be integer.')

        with self.assertRaises(SystemExit) as cm:
            load_data(n_years_test="1")
        self.assertEqual(cm.exception.args[0], '`n_years_val` and `n_years_test` must be integer.')

        # Tests for checks for `splitting_method`
        with self.assertRaises(SystemExit) as cm:
            load_data(splitting_method='unknown_method')
        self.assertEqual(cm.exception.args[0], "`splitting_method` must be 'sequential' or 'random'.")

        # Tests for checks for `season`
        with self.assertRaises(SystemExit) as cm:
            load_data(season='unknown_season')
        self.assertEqual(cm.exception.args[0], "`season` must be None, 'winter' or 'summer'.")

        # Tests for checks for `shuffle_train_data`
        with self.assertRaises(SystemExit) as cm:
            load_data(shuffle_train_data='not_a_bool')
        self.assertEqual(cm.exception.args[0], "`shuffle_train_data` must be a boolean.")

        with self.assertRaises(SystemExit) as cm:
            load_data(shuffle_train_data=1)
        self.assertEqual(cm.exception.args[0], "`shuffle_train_data` must be a boolean.")

        # Tests for checks for `path_to_data`
        with self.assertRaises(SystemExit) as cm:
            load_data(path_to_data=list())
        self.assertEqual(cm.exception.args[0], "`path_to_data` must be a string.")

        with self.assertRaises(SystemExit) as cm:
            load_data(path_to_data='test_file.csv')
        self.assertEqual(cm.exception.args[0], "Can only load netcdf files with file extension '.nc'.")

        with self.assertRaises(SystemExit) as cm:
            load_data(path_to_data='unknown_path.nc')
        self.assertEqual(cm.exception.args[0], "`path_to_data` does not exists.")

    def test_output_format(self):
        """Test transform_to tensor function for type of objects and consistency in methods.
        """
        for key in ['image', 'video', 'sliding_video']:
            # Iterate through all types of Data Loading ...
            length_tot = (364, 40, 40)  # amount of images in train test and validation set
            for set in [0, 1, 2]:
                # ...and though train test an validation set
                length = length_tot[set]
                if key == 'image':
                    Tensor = self.img[set]
                elif key == 'video':
                    Tensor = self.vid[set]
                    length = int(length / self.time_depth)
                else:
                    Tensor = self.sld_vid[set]
                    length = length - self.time_depth + 1

                self.assertIsInstance(Tensor, DataLoader)
                self.assertEqual(Tensor.__len__(), length)


if __name__ == '__main__':
    unittest.main()
