from model import *
from data import *

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(4, 'data/membrane/vol', 'image', 'label', data_gen_args, save_to_dir = None)

model = unet()
#model.summary(())

model_checkpoint = ModelCheckpoint('unet_gelass.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=30, epochs=10, callbacks=[model_checkpoint])


# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("data/membrane/result", results)