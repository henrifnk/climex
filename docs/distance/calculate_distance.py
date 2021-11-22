import docs.distance.sample as sample
import numpy as np
from tqdm import tqdm


def calculate(label, mslp):
    """

    :param label: input label, it can be xarray or list.
    :param mslp: input data, it is mslp or z500 or feature map from nn model. it can be xarray or list.
    :return: return output label, label's corresponding index in input data, and also the distance between each of data.
    """
    label, data_index, mslp = sample.sample(label, mslp)
    mslp = np.array(mslp)
    distance = []
    for i in tqdm(range(len(mslp))):
        calc_mslp = []
        for j in range(len(mslp)):
            if j <= i:
                calc_mslp.append(np.nan)
            if j > i:
                calc = round(np.sum(abs(mslp[j] - mslp[i])) / 10)
                calc_mslp.append(calc)
        distance.append(calc_mslp)
    return label, data_index, distance
