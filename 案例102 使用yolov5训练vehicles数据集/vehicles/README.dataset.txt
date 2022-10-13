# Vehicles-OpenImages > 416x416
https://public.roboflow.ai/object-detection/vehicles-openimages

Provided by [Jacob Solawetz](https://roboflow.ai)
License: CC BY 4.0

![Image example](https://i.imgur.com/ztezlER.png)

# Overview 

This dataset contains 627 images of various vehicle classes for object detection. These images are derived from the [Open Images](https://arxiv.org/pdf/1811.00982.pdf) open source computer vision datasets.

This dataset only scratches the surface of the Open Images dataset for vehicles!

![Image example](https://i.imgur.com/4ZHN8kk.png)

# Use Cases

* Train object detector to differentiate between a car, bus, motorcycle, ambulance, and truck.
* Checkpoint object detector for autonomous vehicle detector
* Test object detector on high density of ambulances in vehicles
* Train ambulance detector
* Explore the quality and range of Open Image dataset

# Tools Used to Derive Dataset

![Image example](https://i.imgur.com/1U0M573.png)

These images were gathered via the [OIDv4 Toolkit](https://github.com/EscVM/OIDv4_ToolKit) This toolkit allows you to pick an object class and retrieve a set number of images from that class **with bound box lables**. 

We provide this dataset as an example of the ability to query the OID for a given subdomain. This dataset can easily be scaled up - please reach out to us if that interests you. 


