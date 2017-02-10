# coding: utf-8

import numpy as np
from sklearn import preprocessing

data = np.array(
    [
        [3, -1.5, 2, -5.4],
        [0, 4, -0.3, 2.1],
        [1, 3.3, -1.9, -4.3]
    ]
)

# mean removal(去除均值)
data_standardized = preprocessing.scale(data)
print ("\nStandardize(标准化) =", data_standardized)
print ("\nMean(均值) =", data_standardized.mean(axis=0))
print ("Std deviation(标准方差) =", data_standardized.std(axis=0))

# min max scaling(极值标准化法)
# 参数 feature_range 表示最小, 最大值
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print ("\nMin max scaled data(极值标准化):\n", data_scaled)

# normalization(正则化)
data_normalized1 = preprocessing.normalize(data, norm='l1')
data_normalized2 = preprocessing.normalize(data, norm='l2')
print ("\nL1 normalized data1:\n", data_normalized1)
print ("\nL2 normalized data2:\n", data_normalized2)

# binarization(二值化)
# threshold 的意思是: 阀值(小于时设置为0, 大于时设置为1)
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print ("\nBinarized data(二值化):\n", data_binarized)

# one hot encoding(独热编码或一位有效编码)
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print ("\nEncoded vector(独热编码):\n", encoded_vector)
