# Project on plant disease prediction 

# ================================================================================
# bringing the dataset into the vs code 
# ================================================================================

import zipfile
from pathlib import Path

home_path = Path.home() / "desktop"

# creating a custom dataset folder to bring our data
custom_dataset = Path.home() / "Project-02" / "Custom_Dataset"
custom_dataset.mkdir(parents= True, exist_ok= True)

# now bringing our dataset images from zip folder 
zip_path = home_path / "data.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(custom_dataset)

# Now we have our dataset in the custom dataset folder
image_dir = custom_dataset / "raw" / "color"
image_paths = list(image_dir.glob("*/*"))

# Now displaying the random images to know more about our dataset
import random
import matplotlib.pyplot as plt
print(f"Total Number of Images: {len(image_paths)}")
random_image = random.choice(image_paths)
a = plt.imread(random_image)
plt.imshow(a)
print(a.shape)
plt.axis("off")
plt.show()


# ==========================================================================
# setting up the data to do the transformations
# ==========================================================================


# creating the transformations for the training and testing data 
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor()
])

# loading the dataset from directory path to real train and test data
from torchvision import datasets
full_train = datasets.ImageFolder(
    root=str(image_dir),
    transform=train_transform
)

full_val = datasets.ImageFolder(
    root=str(image_dir),
    transform=val_transform
)

full_test = datasets.ImageFolder(
    root=str(image_dir),
    transform= test_transform
)


# ===========================================================================================
# now using the random split to divide the data into train , validation and testing dataset
# ===========================================================================================


from torch.utils.data import random_split
import torch

train_size = int(len(image_paths) * 0.7)

val_size = int((len(image_paths)*0.15))

test_size = int(len(image_paths) - train_size - val_size)

generator = torch.Generator().manual_seed(42)
train_data, val_data, test_data = random_split(full_train, 
                                               [train_size, val_size, test_size],
                                               generator=generator)

val_data = torch.utils.data.Subset(full_val, val_data.indices)

test_data = torch.utils.data.Subset(full_test, test_data.indices)

# =======================================================================
# Knowing more about our dataset
# =======================================================================

# printing the random images from the training dataset
image, label = random.choice(train_data)
classes = full_train.classes
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title(classes[label])
plt.show()

# getting a total of ten images randomly with their names
for i in range(10): 
    plt.subplot(2, 5, i+1)
    image, label = random.choice(train_data)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.title(classes[label])

plt.show()

print(f"The number of images in the Training data: {len(train_data)}")
print(f"The number of images in the Validation data: {len(val_data)}")
print(f"The number of images in the Testing data: {len(test_data)}")
print(f"The number of classes are: {len(classes)}")
print(f"The types of classes are: {classes}")
print(f"There are around {len(full_train)/len(classes)} images per class")
print(f"The shape of image is: {image.shape}")

# ==============================================================
# Creating the device augnostic code 
# ==============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# Now creaating the dataloaders and dividing them into batches
# =============================================================

from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_data,
    batch_size=128,
    shuffle=False
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=128,
    shuffle=False
)

# =========================================================
# Creating a simple CNN base model 
# =========================================================

from torch import nn
class base_model(nn.Module):
    def __init__(self,
                 input_units: int,
                 hidden_units: int,
                 output_units: int):
        super().__init__()
        self.Conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels= input_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                              stride= 1),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
            )

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*8*8,
                      out_features= output_units)
        )

    def forward(self, x):
        x = self.Conv_block_1(x)
        x = self.Conv_block_2(x)
        x = self.Conv_block_3(x)
        x = self.Classifier(x)
        return x

# creating the instance of the baseline model 
model = base_model(input_units=3,
                   hidden_units=64,
                   output_units=len(classes)).to(device)

# checking the shape afer every batch in the model in conv blocks
new_image, new_label = next(iter(train_dataloader))
x = new_image
x = x.to(device)
with torch.inference_mode():
    x = model.Conv_block_1(x)
    print(f"Shape after Conv block 1: {x.shape}")
    x = model.Conv_block_2(x)
    print(f"Shape after Conv block 2: {x.shape}")
    x = model.Conv_block_3(x)
    print(f"Shape after Conv bloack 3: {x.shape}")
    x = model.Classifier(x)
    print(f"Shape after Classifier: {x.shape}")


# =========================================================================
# creating the functions to train and validate the model 
# =========================================================================


# creating an accuracy function to calculate the accuracy for later use 
def accuracy_score(actual,
                   predicted):
    acc = (actual == predicted).sum()/len(actual)
    return acc

# Now creating another function to calculate the total time taken by the function 
def calculate_total_time(start, end):
    total_time = end - start
    print(f"The total time taken by {device} is {total_time:.3f} seconds")
    return total_time

# now creating the train function to train the model and test the accuracy 
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device):
    model.train()
    total_acc = 0
    total_loss = 0

    for batch, (x, y) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)
        
        y_logits = model(x)
        y_preds = torch.argmax(y_logits, dim=1)
        
        loss = loss_fn(y_logits, y)
        total_loss += loss.item()
        
        acc = accuracy_score(y, y_preds)
        total_acc += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(dataloader)
    avg_train_acc = total_acc / len(dataloader)

    return avg_train_loss, avg_train_acc

# now comes the validation function 
def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            y_logits = model(x)
            y_preds = torch.argmax(y_logits, dim=1)

            loss = loss_fn(y_logits, y)
            total_loss += loss.item()

            acc = accuracy_score(y, y_preds)
            total_acc += acc.item()

    avg_val_loss = total_loss / len(dataloader)
    avg_val_acc = total_acc / len(dataloader)

    return avg_val_loss, avg_val_acc

# Now creating an overall train function to use both train and test step at the same time 
from tqdm import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          Epoch: int,
          device: torch.device):

    result = {
        "Train_loss" : [],
        "Val_loss" : [],
        "Train_acc" : [],
        "Val_acc" : [],
        "Epochs" : []
    }
    print("Number of training batches:", len(train_dataloader))

    # applying early stopping to the validation dataset 

    patience = 3
    patience_counter = 0
    best_val_loss = float("inf")

    for epoch in tqdm(range(Epoch)):
        print(f"\nEpoch {epoch + 1}/{Epoch}")

        avg_train_loss, avg_train_acc = train_step(model = model,
                                                    dataloader= train_dataloader,
                                                    loss_fn= loss_fn,
                                                    optimizer= optimizer,
                                                    device= device)

        avg_val_loss, avg_val_acc = val_step(model= model,
                                                dataloader= val_dataloader,
                                                loss_fn= loss_fn,
                                                device= device)

        # Adding an early stopping function to stop at the right time

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break


        result["Train_loss"].append(avg_train_loss)
        result["Val_loss"].append(avg_val_loss)
        result["Train_acc"].append(avg_train_acc)
        result["Val_acc"].append(avg_val_acc)
        result["Epochs"].append(epoch+1)

        print(f"Training Loss: {avg_train_loss:.3f}  |   Validation Loss: {avg_val_loss:.3f}")
        print(f"Training Acc: {avg_train_acc:.3f} |    Validation Acc: {avg_val_acc:.3f}")

    return result


# ============================================================================
# Now creating a loss function and an optimizer for the base model 
# ============================================================================

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                             lr = 0.001)

# ============================================================================
# Now training the model and printing the result 
# ============================================================================

from timeit import default_timer as timer

if device == "cuda":
    torch.cuda.synchronize()

start = timer()
print("Model 1:")
result_1 = train(model= model,
            train_dataloader= train_dataloader,
            val_dataloader= val_dataloader,
            loss_fn= loss_function,
            optimizer= optimizer,
            Epoch= 10,
            device= device)

if device == "cuda":
    torch.cuda.synchronize()

end = timer()

calculate_total_time(start= start,
                     end= end)

# =================================================================================
# Now creating the plot of training and validation curve vs epochs
# =================================================================================

plt.subplot(1, 2, 1)
plt.plot(result_1["Epochs"], result_1["Train_loss"], label= "Train Loss")
plt.plot(result_1["Epochs"], result_1["Val_loss"], label= "Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(result_1["Epochs"], result_1["Train_acc"], label= "Train Acc")
plt.plot(result_1["Epochs"], result_1["Val_acc"], label= "Val Acc")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.show()

# ==============================================================================
# improving the cnn model by adding batch Norm, dropout and better architecture
# ==============================================================================

# creating another model by adding CNN + batch Norn + dropout 

class modified_model(nn.Module):
    def __init__(self,
                 input_units: int,
                 hidden_units: int,
                 output_units: int):
        super().__init__()
        self.Conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels= input_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                              stride= 1),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),

            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
            )

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*8*8,
                      out_features= 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_units)
        )

    def forward(self, x):
        x = self.Conv_block_1(x)
        x = self.Conv_block_2(x)
        x = self.Conv_block_3(x)
        x = self.Classifier(x)
        return x

model_2 = modified_model(input_units= 3,
                         hidden_units= 64,
                         output_units= len(classes)).to(device= device)

optimizer_2 = torch.optim.Adam(params= model_2.parameters(),
                                  lr = 0.001)


# creating another model with different better architecture

class better_architecture_model(nn.Module):
    def __init__(self,
                 input_units: int,
                 output_units: int):
        super().__init__()
        self.Conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels= input_units,
                      out_channels= 32,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p = 0.2),

            nn.Conv2d(in_channels= 32,
                      out_channels= 64,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels= 64,
                      out_channels= 128,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p =0.2),

            nn.Conv2d(in_channels= 128,
                      out_channels= 128,
                      padding= 1,
                      kernel_size= 3,
                              stride= 1),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
        )

        self.Conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels= 128,
                      out_channels= 256,
                      padding= 1,
                      stride= 1,
                      kernel_size= 3),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p = 0.2),

            nn.Conv2d(in_channels= 256,
                      out_channels= 256,
                      padding= 1,
                      kernel_size= 3,
                      stride= 1),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(stride= 2,
                         kernel_size= 2)
            )

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= 256*8*8,
                      out_features= 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_units)
        )

    def forward(self, x):
        x = self.Conv_block_1(x)
        x = self.Conv_block_2(x)
        x = self.Conv_block_3(x)
        x = self.Classifier(x)
        return x


model_3 = better_architecture_model(input_units=3,
                                    output_units= len(classes)
                                    ).to(device= device)

optimizer_3 = torch.optim.Adam(params= model_3.parameters(),
                               lr = 0.001)

# =============================================================================
# Now calling all the second and third model and printing the result
# =============================================================================

start_2 = timer()

if device == "cuda":
    torch.cuda.synchronize()

print("Model 2")
result_2 = train(model= model_2,
                 train_dataloader= train_dataloader,
                 val_dataloader= val_dataloader,
                 optimizer= optimizer_2,
                 loss_fn= loss_function,
                 Epoch= 10,
                 device= device)

if device == "cuda":
    torch.cuda.synchronize()

end_2 = timer()
calculate_total_time(start=start_2,
                     end= end_2)

if device == "cuda":
    torch.cuda.synchronize()
 
start_3 = timer()

print("Model 3")
result_3 = train(model= model_3,
                 train_dataloader= train_dataloader,
                 val_dataloader= val_dataloader,
                 optimizer= optimizer_3,
                 loss_fn= loss_function,
                 Epoch= 10,
                 device= device)

if device == "cuda":
    torch.cuda.synchronize()

end_3 = timer()
calculate_total_time(start= start_3,
                     end= end_3)

# ========================================================================
# Now making graphs to check the best model
# ========================================================================

plt.subplot(2, 2, 1)
plt.plot(result_1["Epochs"], result_1["Train_loss"], label = "Model 1")
plt.plot(result_2["Epochs"], result_2["Train_loss"], label = "Model 2")
plt.plot(result_3["Epochs"], result_3["Train_loss"], label = "Model 3")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train Loss")

plt.subplot(2, 2, 2)
plt.plot(result_1["Epochs"], result_1["Val_loss"], label = "Model 1")
plt.plot(result_2["Epochs"], result_2["Val_loss"], label = "Model 2")
plt.plot(result_3["Epochs"], result_3["Val_loss"], label = "Model 3")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Val Loss")

plt.subplot(2, 2, 3)
plt.plot(result_1["Epochs"], result_1["Train_acc"], label = "Model 1")
plt.plot(result_2["Epochs"], result_2["Train_acc"], label = "Model 2")
plt.plot(result_3["Epochs"], result_3["Train_acc"], label = "Model 3")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Train Accuracy")

plt.subplot(2, 2, 4)
plt.plot(result_1["Epochs"], result_1["Val_acc"], label = "Model 1")
plt.plot(result_2["Epochs"], result_2["Val_acc"], label = "Model 2")
plt.plot(result_3["Epochs"], result_3["Val_acc"], label = "Model 3")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Val Accuracy")

plt.tight_layout()
plt.show()

# as we can see that the best model overall by watching the performance is the Second model 

# ============================================================================================
# Now introducing transfer learinig and using resnet model and compairing it with our model 
# ============================================================================================

from torchvision import models
from torchvision.models import resnet18
from torchvision.models import vgg16
from torchvision.models import mobilenet_v3_large

# bringing the resnet model with its weight
resnet_model = models.resnet18(
    weights = models.ResNet18_Weights.DEFAULT
)

# freezing the pretrained layers
for params in resnet_model.parameters():
    params.requires_grad = False

resnet_model.fc = nn.Linear(
    resnet_model.fc.in_features,
    len(classes)
)

resnet_model = resnet_model.to(device=device)

resnet_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, resnet_model.parameters()),
    lr= 0.0001
)

# Nor for the mobilenet model 
mobilenet_model = models.mobilenet_v3_large(
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
)

for params in mobilenet_model.parameters():
    params.requires_grad = False

mobilenet_model.classifier[3] = nn.Linear(
    mobilenet_model.classifier[3].in_features,
    len(classes)
)

mobilenet_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, mobilenet_model.parameters()),
    lr = 0.0001
)

mobilenet_model = mobilenet_model.to(device= device)

# Now for the vgg16 model 
vgg_model = models.vgg16(
    weights = models.VGG16_Weights.DEFAULT
)

for params in vgg_model.parameters():
    params.requires_grad = False

vgg_model.classifier[6] = nn.Linear(
    vgg_model.classifier[6].in_features,
    len(classes)
)

vgg_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, vgg_model.parameters()),
    lr = 0.0001
)

vgg_model = vgg_model.to(device= device)

# As we got all the three pretrained models , now calling them and storing the results in the result dictionary

if device == "cuda":
    torch.cuda.synchronize()

resnet_start = timer()

print("Resnet18 Model: ")
resnet_result = train(model= resnet_model,
                      train_dataloader= train_dataloader,
                      val_dataloader= val_dataloader,
                      optimizer= resnet_optimizer,
                      loss_fn= loss_function,
                      device= device,
                      Epoch= 20)

resnet_end = timer()
calculate_total_time(start= resnet_start,
                     end= resnet_end)

if device == "cuda":
    torch.cuda.synchronize()

mobinet_start = timer()

print("MobileNetV3 Model: ")
mobilenet_result = train(model= mobilenet_model,
                         train_dataloader= train_dataloader,
                         val_dataloader= val_dataloader,
                         optimizer= mobilenet_optimizer,
                         loss_fn= loss_function,
                         device= device,
                         Epoch= 20)
mobinet_end = timer()

calculate_total_time(start= mobinet_start,
                     end= mobinet_end)

if device == "cuda":
    torch.cuda.synchronize()

vgg_start = timer()

print("Vgg16 Model: ")
vgg_result = train(model= vgg_model,
                   train_dataloader=train_dataloader,
                   val_dataloader= val_dataloader,
                   optimizer= vgg_optimizer,
                   loss_fn= loss_function,
                   device= device,
                   Epoch= 20)

vgg_end = timer()
calculate_total_time(start= vgg_start,
                     end= vgg_end)

# Now compairing these results with the best model which is our first model 

plt.subplot(2, 2, 1)
plt.plot(result_2["Epochs"], result_2["Train_loss"], label = "baseline Model")
plt.plot(resnet_result["Epochs"], resnet_result["Train_loss"], label = "Resnet16")
plt.plot(mobilenet_result["Epochs"], mobilenet_result["Train_loss"], label = "Mobile Net v3")
plt.plot(vgg_result["Epochs"], vgg_result["Train_loss"], label = "Vgg16")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(result_2["Epochs"], result_2["Val_loss"], label = "Second Model")
plt.plot(resnet_result["Epochs"], resnet_result["Val_loss"], label = "Resnet16")
plt.plot(mobilenet_result["Epochs"], mobilenet_result["Val_loss"], label = "Mobile Net v3")
plt.plot(vgg_result["Epochs"], vgg_result["Val_loss"], label = "Vgg16")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Val Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(result_2["Epochs"], result_2["Train_acc"], label = "Second Model")
plt.plot(resnet_result["Epochs"], resnet_result["Train_acc"], label = "Resnet16")
plt.plot(mobilenet_result["Epochs"], mobilenet_result["Train_acc"], label = "Mobile Net v3")
plt.plot(vgg_result["Epochs"], vgg_result["Train_acc"], label = "Vgg16")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train Acc")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(result_2["Epochs"], result_2["Val_acc"], label = "Second Model")
plt.plot(resnet_result["Epochs"], resnet_result["Val_acc"], label = "Resnet16")
plt.plot(mobilenet_result["Epochs"], mobilenet_result["Val_acc"], label = "Mobile Net v3")
plt.plot(vgg_result["Epochs"], vgg_result["Val_acc"], label = "Vgg16")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Val Accuracy")
plt.legend()

# So on compairing the result with the best model, the result comes out that the second model is the best model at all 

# ============================================================================
# Now compairing the model 2 with the different learning rates to get the best 
# ============================================================================

# as we can see that the model 2 gives us the best results from all the three models so...

New_optimizer_1 = torch.optim.Adam(model_2.parameters(),
                                   lr = 0.0003)

New_optimizer_2 = torch.optim.Adam(model_2.parameters(),
                                   lr = 0.0001)

New_optimizer_3 = torch.optim.Adam(model_2.parameters(),
                                   lr = 0.00001)


# ===================================================================
# calling all the optimizers with the second model 
# ===================================================================

print("Second Model with learining rate 0.0003")
New_result_1 = train(model= model_2,
                     train_dataloader = train_dataloader,
                     val_dataloader = val_dataloader,
                     optimizer = New_optimizer_1,
                     loss_fn= loss_function,
                     device = device,
                     Epoch = 20)

print("Second Model with learining rate 0.0001")

New_result_2 = train(model = model_2,
                     train_dataloader = train_dataloader,
                     val_dataloader = val_dataloader,
                     optimizer = New_optimizer_2,
                     loss_fn = loss_function,
                     device = device,
                     Epoch = 20)

print("Second Model with learining rate 0.00001")

New_result_3 = train(model= model_2,
                     train_dataloader = train_dataloader,
                     val_dataloader = val_dataloader,
                     optimizer = New_optimizer_3,
                     loss_fn= loss_function,
                     device = device,
                     Epoch = 20)

# ========================================================================
# Now compairing the results of all the four optimizers using the graph
# ========================================================================

plt.subplot(2, 2, 1)
plt.plot(result_2["Epochs"], result_2["Train_loss"], label = "lr: 0.001")
plt.plot(New_result_1["Epochs"], New_result_1["Train_loss"], label = "lr: 0.0003")
plt.plot(New_result_2["Epochs"], New_result_2["Train_loss"], label = "lr: 0.0001")
plt.plot(New_result_3["Epochs"], New_result_3["Train_loss"], label = "lr: 0.00001")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(result_2["Epochs"], result_2["Val_loss"], label = "lr: 0.001")
plt.plot(New_result_1["Epochs"], New_result_1["Val_loss"], label = "lr: 0.0003")
plt.plot(New_result_2["Epochs"], New_result_2["Val_loss"], label = "lr: 0.0001")
plt.plot(New_result_3["Epochs"], New_result_3["Val_loss"], label = "lr: 0.00001")
plt.title("Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(result_2["Epochs"], result_2["Train_acc"], label = "lr: 0.001")
plt.plot(New_result_1["Epochs"], New_result_1["Train_acc"], label = "lr: 0.0003")
plt.plot(New_result_2["Epochs"], New_result_2["Train_acc"], label = "lr: 0.0001")
plt.plot(New_result_3["Epochs"], New_result_3["Train_acc"], label = "lr: 0.00001")
plt.title("Train Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(result_2["Epochs"], result_2["Val_acc"], label = "lr: 0.001")
plt.plot(New_result_1["Epochs"], New_result_1["Val_acc"], label = "lr: 0.0003")
plt.plot(New_result_2["Epochs"], New_result_2["Val_acc"], label = "lr: 0.0001")
plt.plot(New_result_3["Epochs"], New_result_3["Val_acc"], label = "lr: 0.00001")
plt.title("Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()

plt.tight_layout()
plt.show()

# as the on compairing all the models, the second model is the best model with learining rate 0.00001 so far.....
# so creating the best model again to get the fresh weights

best_model = modified_model(
    input_units=3,
    hidden_units=64,
    output_units=len(classes)
).to(device)

best_optimizer = torch.optim.Adam(
    params=best_model.parameters(),
    lr=0.00001
)

print("Best Model: ")
best_result = train(model= best_model,
                    train_dataloader= train_dataloader,
                    val_dataloader= val_dataloader,
                    loss_fn= loss_function,
                    optimizer= best_optimizer,
                    device= device,
                    Epoch= 20)

# ====================================================================
# Now creating a function for our test dataloader 
# ====================================================================

def prediction(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):

    model.eval()
    total_acc = 0
    total_loss = 0
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(test_dataloader):
            x = x.to(device = device)
            y = y.to(device = device)

            y_logits = model(x)
            y_preds = torch.argmax(y_logits, dim= 1)

            loss = loss_fn(y_logits, y)
            total_loss += loss.item()

            acc = accuracy_score(y, y_preds)
            total_acc += acc.item()

        avg_test_loss = total_loss / len(test_dataloader)
        avg_test_acc = total_acc / len(test_dataloader)

    return avg_test_loss, avg_test_acc

# making a pred function to store the results directly into the pred results dictionary

def pred(model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device):

    pred_result = {
        "Test_loss" : [],
        "Test_acc" : [],
    }

    avg_test_loss, avg_test_acc = prediction(model= model,
                                                 test_dataloader= test_dataloader,
                                                 loss_fn= loss_fn,
                                                 device= device)

    pred_result["Test_loss"].append(avg_test_loss)
    pred_result["Test_acc"].append(avg_test_acc)

    print(f"Testing Loss: {pred_result['Test_loss']}")
    print(f"Testing Acc: {pred_result['Test_acc']}")

    return pred_result
    

# Now calling the function with test dataloader created at the start

pred_result = pred(model= best_model,
                   test_dataloader= test_dataloader,
                   loss_fn= loss_function,
                   device= device)

pred_rest_2 = pred(model= model_2,
                   test_dataloader= test_dataloader,
                   loss_fn= loss_function,
                   device= device)

# ================================================================================
# Now creating accuracy, precision, recall, f1-score and confusion matrix 
# ================================================================================

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             classification_report)

import numpy as np

def classification_metrices(model: torch.nn.Module,
                            test_dataloader: torch.utils.data.DataLoader,
                            device: torch.device):
    
    model.eval()
    all_y_preds = []
    all_y_true = []

    with torch.inference_mode():
        for batch, (x, y) in enumerate(test_dataloader):

            x = x.to(device)
            y = y.to(device)

            y_logits = model(x)
            y_preds = torch.argmax(y_logits, dim=1)

            all_y_preds.extend(y_preds.cpu().numpy())
            all_y_true.extend(y.cpu().numpy())

        actual = np.array(all_y_true)
        predicted = np.array(all_y_preds)

        accuracy = accuracy_score(actual, predicted)

        precision = precision_score(actual, predicted, average="macro")

        recall = recall_score(actual, predicted, average="macro")

        f1 = f1_score(actual, predicted, average="macro")

        conf_matrix = confusion_matrix(actual, predicted)

        report = classification_report(actual, predicted)

    return accuracy, precision, recall, f1, conf_matrix, report


print("Classification report of the best model")
classification_metrices(model= best_model,
                        test_dataloader= test_dataloader,
                        device= device)

print("Classification report of the second model")
classification_metrices(model= model_2,
                        test_dataloader= test_dataloader,
                        device= device)


# Now we are all set 

# ==============================================================
# printing the random actual and predicted image by the second model 
# ==============================================================

# choosing a random image from the grayscalsle and segmented folder

grayscale_image_path = custom_dataset / "raw" / "grayscale"
segmentad_image_path = custom_dataset / "raw" / "segmented"

all_images = list(grayscale_image_path.glob("*/*")) + list(segmentad_image_path.glob("*/*"))
random_img = random.choice(all_images)

from PIL import Image

random_img = random.choice(all_images)

img = Image.open(random_img).convert("RGB")
x = val_transform(img).unsqueeze(0).to(device)

best_model.eval()

with torch.inference_mode():
    logits = best_model(x)
    predicted_class = logits.argmax(dim=1).item()

plt.imshow(img)
plt.axis("off")
plt.title(
    f"Actual: {random_img.parent.name} | "
    f"Predicted: {classes[predicted_class]}"
)
plt.show()

# Now predicting 9 random images and plotting them in game graph and testing the model predictions 

print("Images from the best model")
random_images = random.sample(all_images, 9)
best_model.eval()
with torch.inference_mode():
    for i in range(9):
        plt.subplot(3, 3, i+1)

        img = Image.open(random_images[i]).convert("RGB")
        x = val_transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = best_model(x)
            prediction = logits.argmax(dim=1).item()

        if random_images[i].parent.name == classes[prediction]:
            title_color = "green"
        else:
            title_color = "red"

        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Actual: {random_images[i].parent.name} | "
            f"Predicted: {classes[prediction]}"
        )

plt.tight_layout()
plt.show()

# doing same with the second model and compairing it with the best model 

print("Images from the second model")

model_2.eval()

plt.figure(figsize=(12, 12))

with torch.inference_mode():
    for i in range(9):
        img = Image.open(random_images[i]).convert("RGB")
        x = val_transform(img).unsqueeze(0).to(device)

        logits = model_2(x)
        predicted_class = logits.argmax(dim=1).item()

        actual_class = random_images[i].parent.name
        title_color = (
            "green" if actual_class == classes[predicted_class]
            else "red"
        )

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Actual: {actual_class}\n"
            f"Predicted: {classes[predicted_class]}",
            color=title_color,
            fontsize=8
        )

plt.tight_layout()
plt.show()