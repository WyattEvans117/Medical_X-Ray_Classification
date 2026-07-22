# Medical_X-Ray_Classification
## Summary
This repository contains GPU accelerated computer vision pipeline developed in Python with PyTorch to classify medical images in 3 distinct categories: Pneumonia, Tuberculosis, and Normal lungs.
This is trained from a publicly available dataset. While many computer vision projects rely entirely on pre-trained weights, this model was intentionally built and trained from scratch. This is simple to change within the models.resnet18(weights=None).

## Architecture, dataset scope and evaluation metrics
It uses a convolutional neural network (ResNet18), the ResNet18 model provides robust gradient flow while being resilient to overfitting when dealing with specialized smaller medical image datasets.
This project uses GPU processing as opposed to CPU processing which drastically cuts compute time.
The dataset is pulled via API programming so as to avoid locally downloading a large dataset.
I utilize stochastic gradient descent in the program (SGD) which is known to get more and more accurate as you increase the amount of epochs (convergence). 
Presently the code is a "mini-batch SGD" for the sake of time complexity.
For the sake of immediate demonstration and rapid prototyping the data volume and epoch count are intentionally restricted to a streamlined subset.
In terms of model evaluation, an overall score and an F-1 score is generated along with a confusion matrix to discover class imbalances. 
