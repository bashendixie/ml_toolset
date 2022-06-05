# import the necessary packages
import mxnet as mx

class MxGoogLeNet:
    @staticmethod
    def conv_module(data, K, kX, kY, pad=(0, 0), stride=(1, 1)):
        # define the CONV => BN => RELU pattern
        conv = mx.sym.Convolution(data=data, kernel=(kX, kY), num_filter=K, pad=pad, stride=stride)
        bn = mx.sym.BatchNorm(data=conv)
        act = mx.sym.Activation(data=bn, act_type="relu")

        # return the block
        return act

    @ staticmethod
    def inception_module(data, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj):
        # the first branch of the Inception module consists of 1x1
        # convolutions
        conv_1x1 = MxGoogLeNet.conv_module(data, num1x1, 1, 1)

        # the second branch of the Inception module is a set of 1x1
        # convolutions followed by 3x3 convolutions
        conv_r3x3 = MxGoogLeNet.conv_module(data, num3x3Reduce, 1, 1)
        conv_3x3 = MxGoogLeNet.conv_module(conv_r3x3, num3x3, 3, 3, pad=(1, 1))

        # the third branch of the Inception module is a set of 1x1
        # convolutions followed by 5x5 convolutions
        conv_r5x5 = MxGoogLeNet.conv_module(data, num5x5Reduce, 1, 1)
        conv_5x5 = MxGoogLeNet.conv_module(conv_r5x5, num5x5, 5, 5, pad = (2, 2))

        # the final branch of the Inception module is the POOL +
        # projection layer set
        pool = mx.sym.Pooling(data=data, pool_type="max", pad=(1, 1), kernel = (3, 3), stride = (1, 1))
        conv_proj = MxGoogLeNet.conv_module(pool, num1x1Proj, 1, 1)

        # concatenate the filters across the channel dimension
        concat = mx.sym.Concat(*[conv_1x1, conv_3x3, conv_5x5, conv_proj])
        # return the block
        return concat

    @staticmethod
    def build(classes):
        # data input
        data = mx.sym.Variable("data")

        # Block #1: CONV => POOL => CONV => CONV => POOL
        conv1_1 = MxGoogLeNet.conv_module(data, 64, 7, 7, pad=(3, 3), stride=(2, 2))
        pool1 = mx.sym.Pooling(data=conv1_1, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
        conv1_2 = MxGoogLeNet.conv_module(pool1, 64, 1, 1)
        conv1_3 = MxGoogLeNet.conv_module(conv1_2, 192, 3, 3, pad=(1, 1))
        pool2 = mx.sym.Pooling(data=conv1_3, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Block #3: (INCEP * 2) => POOL
        in3a = MxGoogLeNet.inception_module(pool2, 64, 96, 128, 16, 32, 32)
        in3b = MxGoogLeNet.inception_module(in3a, 128, 128, 192, 32, 96, 64)
        pool3 = mx.sym.Pooling(data=in3b, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Block #4: (INCEP * 5) => POOL
        in4a = MxGoogLeNet.inception_module(pool3, 192, 96, 208, 16,48, 64)
        in4b = MxGoogLeNet.inception_module(in4a, 160, 112, 224, 24,64, 64)
        in4c = MxGoogLeNet.inception_module(in4b, 128, 128, 256, 24,64, 64)
        in4d = MxGoogLeNet.inception_module(in4c, 112, 144, 288, 32,64, 64)
        in4e = MxGoogLeNet.inception_module(in4d, 256, 160, 320, 32,128, 128)
        pool4 = mx.sym.Pooling(data=in4e, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))

        # Block #5: (INCEP * 2) => POOL => DROPOUT
        in5a = MxGoogLeNet.inception_module(pool4, 256, 160, 320, 32, 128, 128)
        in5b = MxGoogLeNet.inception_module(in5a, 384, 192, 384, 48, 128, 128)
        pool5 = mx.sym.Pooling(data=in5b, pool_type="avg", kernel=(7, 7), stride=(1, 1))
        do = mx.sym.Dropout(data=pool5, p=0.4)

        # softmax classifier
        flatten = mx.sym.Flatten(data=do)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc1, name="softmax")
        # return the network architecture
        return model


