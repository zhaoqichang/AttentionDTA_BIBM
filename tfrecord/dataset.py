import numpy as np
import json
import os
import tensorflow as tf

CHARPROTSET = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5,
               "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
               "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17,
               "W": 18, "X": 19, "Y": 20}

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64


def label_smiles(line):
    X = np.ones(MAX_SMI_LEN, dtype=np.int64) * 66
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = CHARISOSMISET[ch]
    return X  # .tolist()

def label_sequence(line):
    X = np.ones(MAX_SEQ_LEN, dtype=np.int64) * 21
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X  # .tolist()


def convertToGraph(line):
    field = line.split()
    # smi_name = field[0]
    # target_name = field[1]
    smi = field[2]
    protien = field[3]
    affinity = float(field[4])
    drug = label_smiles(smi)
    words = label_sequence(protien)
    drug_emded = tf.train.Feature(
        int64_list=tf.train.Int64List(value=drug))
    protein_emded = tf.train.Feature(
        int64_list=tf.train.Int64List(value=words))
    Affinity = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[affinity]))
    DTA_dict = {
        'drug': drug_emded,
        'protein': protein_emded,
        'affinity': Affinity
    }
    # feed dictionary to tf.train.Features
    tf_features = tf.train.Features(feature=DTA_dict)
    # get an example
    tf_example = tf.train.Example(features=tf_features)
    # serialize the example
    tf_serialized = tf_example.SerializeToString()
    return tf_serialized


for shujuji in range(2):
    if shujuji == 0:
        dbName = "kiba"  # kiba,davis,test
    else:
        dbName = "davis"
    if os.path.exists(dbName) is False:
        os.makedirs(dbName)

    MAX_SEQ_LEN = 1200
    MAX_SMI_LEN = 100

    data_fold = json.load(open("./" + dbName + "_div.txt"))
    fold = []
    train_fold = []
    test_fold = data_fold
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            else:
                fold.extend(data_fold[j])
        train_fold.append(fold)
        fold = []

    for i in range(5):
        print("begin %s:%d" % (dbName, i))
        train_name = "./" + dbName + "/train%d.tfrecord" % i
        test_name = "./" + dbName + "/test%d.tfrecord" % i
        with tf.python_io.TFRecordWriter(train_name) as trainwriter, \
                tf.python_io.TFRecordWriter(test_name) as testwriter:
            with open('./' + dbName + '_str_all.txt', 'r') as file_txt:
                file_txt = file_txt.readlines()
                print("begin train_set:%d" % i)
                for line_num in train_fold[i]:
                    line = file_txt[line_num]
                    tf_serialized = convertToGraph(line)
                    # write
                    trainwriter.write(tf_serialized)
                print("begin test_set:%d" % i)
                for line_num in test_fold[i]:
                    line = file_txt[line_num]
                    tf_serialized = convertToGraph(line)
                    # write
                    testwriter.write(tf_serialized)
                print("Finish %d" % i)
