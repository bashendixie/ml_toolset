# import the necessary packages
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model
import matplotlib.pyplot as plt


def get_train_monitor(testDs, outputPath, stnLayerName):
    # iterate over the test dataset and take a batch of test images
    (testImg, _) = next(iter(testDs))

    # define a training monitor
    class TrainMonitor(Callback):
        def on_epoch_end(self, epoch, logs=None):
            model = Model(self.model.input,
                          self.model.get_layer(stnLayerName).output)
            testPred = model(testImg)
            # plot the image and the transformed image
            _, axes = plt.subplots(nrows=5, ncols=2, figsize=(5, 10))
            for ax, im, t_im in zip(axes, testImg[:5], testPred[:5]):
                ax[0].imshow(im[..., 0], cmap="gray")
                ax[0].set_title(epoch)
                ax[0].axis("off")
                ax[1].imshow(t_im[..., 0], cmap="gray")
                ax[1].set_title(epoch)
                ax[1].axis("off")

            # save the figures
            plt.savefig(f"{outputPath}/{epoch:03d}")
            plt.close()

    # instantiate the training monitor callback
    trainMonitor = TrainMonitor()
    # return the training monitor object
    return trainMonitor