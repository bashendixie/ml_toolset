import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader
	dl = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
	# return the data loader
	return dl

def train_val_split(dataset, valSplit=0.2):
	# grab the total size of the dataset
	totalSize = len(dataset)
	# perform training and validation split
	(trainIdx, valIdx) = train_test_split(list(range(totalSize)),
		test_size=valSplit)
	trainDataset = Subset(dataset, trainIdx)
	valDataset = Subset(dataset, valIdx)
	# return training and validation dataset
	return (trainDataset, valDataset)