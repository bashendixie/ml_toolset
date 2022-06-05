# import the necessary packages
from torch.utils.data import DataLoader

def get_dataloader(dataset, batchSize, shuffle=True):
	# 创建一个数据加载器并返回
	dataLoader= DataLoader(dataset, batch_size=batchSize,
		shuffle=shuffle)
	return dataLoader

def normalize(image, mean=128, std=128):
    # 标准化 SSD 输入并返回
    image = (image * 256 - mean) / std
    return image