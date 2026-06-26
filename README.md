# Medical_X-Ray_Classification
A quick computer vision project made to showcase skills related to AI engineering.
This is trained from a publicly available dataset. The model does not use transfer learning as I wanted to investigate training a model from scratch. This is simple to change within the models.resnet18(weights=None).
It uses a convultional neural network (ResNet18), the ResNet model is a bit older but is usually better at smaller datasets.
I utilize stochoastic gradient descent in the program (SGD) which is known to get more and more accurate as you increase the amount of epochs (convergence). 
Presently the code is a "mini-batch SGD" for the sake of time complexity.
I left the data intentionally lower than the whole dataset to speed up testing and showcasing purposes but a more accurate model would likely be created with an increase in the number of training images and epochs.
This is not meant to showcase every skill related to AI but should show a willingness to learn and an ability to apply concepts into code.
