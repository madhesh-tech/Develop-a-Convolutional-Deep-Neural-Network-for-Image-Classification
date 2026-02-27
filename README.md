# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
<img width="907" height="639" alt="image" src="https://github.com/user-attachments/assets/eab62422-3266-45db-a71c-ca39f4313731" />

## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data
### STEP 2: 
Get the shape of the first image in the training dataset


### STEP 3: 

Get the shape of the first image in the test dataset

### STEP 4: 
rain the Model 


### STEP 5: 

Test the Model 6.Predict on a Single Image

### STEP 6: 
Display the image




## PROGRAM

### Name: Madhesh I

### Register Number: 212224220055

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
     
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)





    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:Madhesh I')
print('Register Number:212224220055')
summary(model, input_size=(1, 28, 28))
# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for images,labels in train_loader:
          optimizer.zero_grad()
          outputs=model(images)
          loss=criterion(outputs,labels)
          loss.backward()
          optimizer.step()
          running_loss+=loss.item()


        print('Name:Madhesh I')
        print('Register Number: 212224220055')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
# Train the model
train_model(model, train_loader)
## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:Madhesh I')
    print('Register Number:212224220055')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:Prasanth T')
    print('Register Number:212224040243')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:Madhesh I')
    print('Register Number:212224220055')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
# Evaluate the model
test_model(model, test_loader)
## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name:Madhesh I')
    print('Register Number:212224220055')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
# Example Prediction
predict_image(model, image_index=80, dataset=test_dataset)


```

### OUTPUT

## Training Loss per Epoch

<img width="287" height="192" alt="Screenshot 2026-02-13 180253" src="https://github.com/user-attachments/assets/2a66a0b5-5f75-4ed6-aff8-ee67d4518a39" />

## Confusion Matrix

<img width="1054" height="784" alt="Screenshot 2026-02-13 180614" src="https://github.com/user-attachments/assets/4c9ffbae-3ea6-46e3-b18c-d294c8606996" />

## Classification Report
<img width="579" height="425" alt="Screenshot 2026-02-13 180638" src="https://github.com/user-attachments/assets/b27c2322-8e1b-4d43-906b-4ea65bca8e42" />

### New Sample Data Prediction
<img width="682" height="546" alt="Screenshot 2026-02-13 180735" src="https://github.com/user-attachments/assets/bce3e2ea-5eaa-4ad3-88e8-0d371fd190d9" />

## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully
