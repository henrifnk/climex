import random


def sample(label, mslp):
    """

    :param label: input label, it can be xarray or list.
    :param mslp: input data, like mslp or z500 or feature map from nn model. it can be xarray or list.
    :return: return output label, label's corresponding index in input data, and also the data itself.
    """
    label = label
    mslp = mslp

    data_index_0 = []
    data_index_11 = []
    data_index_17 = []

    for i in range(len(label)):
        if label[i] == 0:
            data_index_0.append(i)
        if label[i] == 11:
            data_index_11.append(i)
        if label[i] == 17:
            data_index_17.append(i)

    random.seed(1)
    if len(data_index_0) >= len(data_index_11) + len(data_index_17):
        sample_0 = random.sample(data_index_0, len(data_index_11) + len(data_index_17))
    else:
        sample_0 = random.sample(data_index_0, len(data_index_0))

    mslp_0 = []

    for i in range(len(sample_0)):
        mslp_0.append(mslp[sample_0[i]])

    random.seed(1)
    sample_11 = random.sample(data_index_11, len(data_index_11))
    mslp_11 = []

    for i in range(len(sample_11)):
        mslp_11.append(mslp[sample_11[i]])

    random.seed(1)
    sample_17 = random.sample(data_index_17, len(data_index_17))
    mslp_17 = []

    for i in range(len(sample_17)):
        mslp_17.append(mslp[sample_17[i]])

    new_label = []

    for i in range(len(mslp_0)):
        new_label.append(0)
    for i in range(len(mslp_11)):
        new_label.append(11)
    for i in range(len(mslp_17)):
        new_label.append(17)

    sample_0.extend(sample_11)
    sample_0.extend(sample_17)

    mslp_0.extend(mslp_11)
    mslp_0.extend(mslp_17)

    return new_label, sample_0, mslp_0
