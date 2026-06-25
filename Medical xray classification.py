import kaggle
import matplotlib.pyplot as plt
import numpy as np
import torch as py
import torchmetrics
import torchvision.models as models
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader #utilized to test outputs
from torchvision.transforms import ToTensor #creates the necessary vectors corresponding to pixel values
kaggle.api.authenticate()
kaggle.api.dataset_download_files("muhammadrehan00/chest-xray-dataset", path=".",unzip=True) #setting up the dataset utilizing API method (need to select input on right)
print(kaggle.api.dataset_list_files("muhammadrehan00/chest-xray-dataset").files) #confirmation that the dataset is loading in
image_resize = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
]) 
#This creates all the images to same size, higher resolution means more accurate image but more GPU usage. Typically stick to multiples of 32.

train_dataset = ImageFolder(
    root="/kaggle/working/train",
    transform=image_resize
)
#loading the data into training set

test_dataset = ImageFolder(
    root="/kaggle/working/test",
    transform=image_resize
)
#Another portion of the data goes into a testing set

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4) 
data_iter = iter(train_loader)
images, labels = next(data_iter)
img= images[0] #checks the training dataset with a random sample of 32 and returns 1 random image
img_numpy = img.permute(1, 2, 0).numpy() #numpy and matplotlib store values differently, must be permutated to match specifications
plt.imshow(img_numpy) #ensuring images are loading correctly
num_classes = 3 #number of classes
task_type = "multiclass"
precision = torchmetrics.Precision(task="multiclass", num_classes=3, average="macro")

predictions = py.tensor([0,2,1,0,2,1])
targets = py.tensor([0,1,1,0,2,2])

macro_precision = precision(predictions, targets)
print(f"Precision: {macro_precision.item(): .4f}") #This code creates a precision score but is generated from a 1D array.
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

predictions = py.tensor([0, 2, 1, 0, 2, 1])
targets     = py.tensor([0, 1, 1, 0, 2, 2])

confusion_matrix_metrics = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=3, normalize=None)
confusion_matrix = confusion_matrix_metrics(predictions, targets)
print(confusion_matrix)


model.eval() 
with py.no_grad():
    for images, targets in train_loader:
        outputs = model(images)
        predictions = py.argmax(outputs, dim=1)
        confusion_matrix_metrics.update(predictions, targets)

final_confusion_matrix = confusion_matrix_metrics.compute()
print("Final 3x3 Confusion Matrix:\n", final_confusion_matrix)

fig, ax = confusion_matrix_metrics.plot(labels=["Normal", "Pneumona", "Tuberculosis"])
plt.title("k confusion matrix")
plt.show
#CURRENTLY SHOW HEAVY BIAS TO RIGHTMOST COLUMN
!ls -l /kaggle/input/