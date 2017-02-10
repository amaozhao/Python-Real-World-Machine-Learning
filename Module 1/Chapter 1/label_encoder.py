# coding: utf-8

# import numpy as np
from sklearn import preprocessing

# 标签编码
label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)

# print classes
print ("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print (item, '-->', i)

# transform a set of classes(分类转换)
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print ("\nLabels =", labels)
print ("Encoded labels =", list(encoded_labels))

# inverse transform(反转)
encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print ("\nEncoded labels =", encoded_labels)
print ("Decoded labels =", list(decoded_labels))
