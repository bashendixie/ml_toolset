from torch.nn import Linear
from torch.nn import Module

class Classifier(Module):
	def __init__(self, baseModel, numClasses, model):
		super().__init__()
		# 初始化基本模型
		self.baseModel = baseModel
		# 检查基础模型是否为 VGG，如果是，则相应地初始化 FC 层
		if model == "vgg":
			self.fc = Linear(baseModel.classifier[6].out_features, numClasses)
		# 否则，基础模型是 ResNet 类型，因此相应地初始化 FC 层
		else:
			self.fc = Linear(baseModel.fc.out_features, numClasses)

	def forward(self, x):
		# 通过基本模型传递输入以获取特征，然后通过全连接层传递特征以获取我们的输出 logits
		features = self.baseModel(x)
		logits = self.fc(features)
		# 返回分类器输出
		return logits