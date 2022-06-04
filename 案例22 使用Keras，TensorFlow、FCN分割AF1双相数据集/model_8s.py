from keras.layers import *
from keras.optimizers import *
from keras.models import *


project_name = "fcn_segment"

channels = 1
std_shape = (256, 256, channels) # 输入尺寸, std_shape[0]: img_rows, std_shape[1]: img_cols
                                  # 这个尺寸按你的图像来, 如果你的图大小不一, 那 img_rows 和 image_cols
                                  # 都要设置成 None, 如果你在用 Generator 加载数据时有扩展边缘, 那 std_shape
                                  # 就是扩展后的尺寸

img_input = Input(shape = std_shape, name = "input")

conv_1 = Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_1")(img_input)
max_pool_1 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_1")(conv_1)

conv_2 = Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_2")(max_pool_1)
max_pool_2 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_2")(conv_2)

conv_3 = Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_3")(max_pool_2)
max_pool_3 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_3")(conv_3)

conv_4 = Conv2D(256, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_4")(max_pool_3)
max_pool_4 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_4")(conv_4)

conv_5 = Conv2D(512, kernel_size = (3, 3), activation = "relu", padding = "same", name = "conv_5")(max_pool_4)
max_pool_5 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "max_pool_5")(conv_5)

# max_pool_5 转置卷积上采样 2 倍和 max_pool_4 一样大
up6 = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = "same", kernel_initializer = "he_normal",
                                   name = "upsamping_6")(max_pool_5)
                
_16s = add([max_pool_4, up6])

# _16s 转置卷积上采样 2 倍和 max_pool_3 一样大
up_16s = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = "same", kernel_initializer = "he_normal",
                                      name = "Conv2DTranspose_16s")(_16s)
                                 
_8s = add([max_pool_3, up_16s])

# _8s 上采样 8 倍后与输入尺寸相同
up7 = UpSampling2D(size = (8, 8), interpolation = "bilinear", name = "upsamping_7")(_8s)

# 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
conv_7 = Conv2D(1, kernel_size = (3, 3), activation = "sigmoid", padding = "same", name = "conv_7")(up7)

model = Model(img_input, conv_7, name = project_name)

model.compile(optimizer=Adam(lr=1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()