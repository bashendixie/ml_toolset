# USAGE
# python inference.py
# import the necessary packages
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-images", type=int, default=64, help="# of images you want the DCGAN to generate")
args = vars(ap.parse_args())
# check if gpu is available for use
useGpu = True if torch.cuda.is_available() else False
# load the DCGAN model
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo-hub", "DCGAN", pretrained=True, useGPU=useGpu, source='local')

# generate random noise to input to the generator
(noise, _) = model.buildNoiseData(args["num_images"])
# turn off autograd and feed the input noise to the model
with torch.no_grad():
	generatedImages = model.test(noise)
# reconfigure the dimensions of the images to make them channel
# last and display the output
output = torchvision.utils.make_grid(generatedImages).permute(1, 2, 0).cpu().numpy()
plt.imshow(output)
plt.show()

