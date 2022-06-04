from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.applications import VGG16

def FCN32(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height, input_width, 3))
    model = VGG16(include_top=False, weights='imagenet',input_tensor=img_input)
    
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    # 内存原因，卷积核4096时报错OOM，降低至1024
    o = Conv2D(filters=1024,kernel_size=(7,7),padding='same',activation='relu',name='fc6')(model.output)
    o = Dropout(0.5)(o)
    o = Conv2D(filters=1024,kernel_size=(1,1),padding='same',activation='relu',name='fc7')(o)
    o = Dropout(0.5)(o)

    o = Conv2D(filters=nClasses,kernel_size=(1,1),padding='same',activation='relu',name='score_fr')(o)
    o = Conv2DTranspose(filters=nClasses,kernel_size=(32,32), strides=(32,32), padding='valid', activation=None, name='score2')(o)
    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn8 = Model(img_input,o)
    fcn8.summary()
    return fcn8


def FCN16(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height,input_width,3))
    # model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    model = FCN32(11, 320, 320)
    model.load_weights("model.h5")


    skip1 = Conv2DTranspose(512,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="upsampling6")(model.get_layer("fc7").output)
    summed = add(inputs=[skip1,model.get_layer("block4_pool").output])
    up7 = UpSampling2D(size=(16,16),interpolation='bilinear',name='upsamping_7')(summed)
    o = Conv2D(nClasses,kernel_size=(3,3),activation='relu',padding='same',name='conv_7')(up7)


    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn16 = Model(model.input,o)
    return fcn16


def FCN8(nClasses,input_height,input_width):
    #model = FCN32(1, 320, 320)
    #model.load_weights("model.h5")

    img_input = Input(shape=(input_height, input_width, 3))
    #model = VGG16(include_top=False, weights='imagenet',input_tensor=img_input)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    # 内存原因，卷积核4096时报错OOM，降低至1024
    #o = Conv2D(filters=1024,kernel_size=(7,7),padding='same',activation='relu',name='fc6')(x)
    #o = Dropout(0.5)(o)
    #fc7 = Conv2D(filters=1024,kernel_size=(1,1),padding='same',activation='relu',name='fc7')(o)
    #fc7 = Dropout(0.5)(fc7)

    skip1 = Conv2DTranspose(512,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="up7")(x)
    summed = add([skip1, block4_pool])
    
    skip2 = Conv2DTranspose(256,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name='up4')(summed)
    summed = add([skip2, block3_pool])

    up7 = UpSampling2D(size=(8, 8), interpolation='bilinear', name='upsamping_7')(summed)
    o = Conv2D(nClasses, kernel_size=(3,3), activation='relu', padding='same', name='conv_7')(up7)

    #o = Reshape((-1, nClasses))(o)
    o = Activation("softmax")(o)
    fcn8 = Model(img_input, o)
    fcn8.compile(optimizer=Adam(lr=1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
    return fcn8

