# AttentionDTA_BIBM
 AttentionDTA: prediction of drug–target binding affinity using attention model.https://ieeexplore.ieee.org/abstract/document/8983125

This repository contains the source code and the data.

## AttentionDTA

![](model.png)

## Setup and dependencies 

Dependencies:
- python 3.6
- tensorflow >=1.9
- numpy

## Resources:
+ README.md: this file.
+ tfrecord: The original data set and data set processing code are saved in this folder.
	+ davis_div.txt: Under the 5-fold cross-validation setting, there is a division of the training set and the test set of the davis data set.
/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data
+ pretrained: models trained by the proposed framework 

##  source codes:
+ create_data.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ predict_with_pretrained_model.py: run this to predict affinity for testing data using models already trained stored at folder pretrained/
+ training.py: train a GraphDTA model.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.
