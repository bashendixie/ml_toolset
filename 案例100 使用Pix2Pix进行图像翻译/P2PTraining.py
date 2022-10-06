# import the necessary packages
from keras import Model
import tensorflow as tf


class Pix2PixTraining(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        # initialize the generator, discriminator
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gOptimizer, dOptimizer, bceLoss, maeLoss):
        super().compile()
        # initialize the optimizers for the generator
        # and discriminator
        self.gOptimizer = gOptimizer
        self.dOptimizer = dOptimizer

        # initialize the loss functions
        self.bceLoss = bceLoss
        self.maeLoss = maeLoss

    def train_step(self, inputs):
        # grab the input mask and corresponding real images
        (inputMask, realImages) = inputs

        # initialize gradient tapes for both generator and discriminator
        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            # generate fake images
            fakeImages = self.generator(inputMask, training=True)

            # discriminator output for real images and fake images
            discRealOutput = self.discriminator([inputMask, realImages], training=True)
            discFakeOutput = self.discriminator([inputMask, fakeImages], training=True)
            # compute the adversarial loss for the generator
            misleadingImageLabels = tf.ones_like(discFakeOutput)
            ganLoss = self.bceLoss(misleadingImageLabels, discFakeOutput)

            # compute the mean absolute error between the fake and the
            # real images
            l1Loss = self.maeLoss(realImages, fakeImages)

            # compute the total generator loss
            totalGenLoss = ganLoss + (10 * l1Loss)

            # discriminator loss for real and fake images
            realImageLabels = tf.ones_like(discRealOutput)
            realDiscLoss = self.bceLoss(realImageLabels, discRealOutput)
            fakeImageLabels = tf.zeros_like(discFakeOutput)
            generatedLoss = self.bceLoss(fakeImageLabels, discFakeOutput)

            # compute the total discriminator loss
            totalDiscLoss = realDiscLoss + generatedLoss

        # calculate the generator and discriminator gradients
        generatorGradients = genTape.gradient(totalGenLoss, self.generator.trainable_variables)
        discriminatorGradients = discTape.gradient(totalDiscLoss, self.discriminator.trainable_variables)

        # apply the gradients to optimize the generator and discriminator
        self.gOptimizer.apply_gradients(zip(generatorGradients, self.generator.trainable_variables))
        self.dOptimizer.apply_gradients(zip(discriminatorGradients, self.discriminator.trainable_variables))

        # return the generator and discriminator losses
        return {"dLoss": totalDiscLoss, "gLoss": totalGenLoss}