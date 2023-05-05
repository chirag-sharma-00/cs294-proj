import timm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
torch.manual_seed(0)

with open("training_log.txt", "w") as f:
    f.write("Epoch \t Training Loss \t Training Accuracy \t Validation Loss \t Validation Accuracy \n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_set = torch.load("dataset.pt")
# 80/20 train/val split
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])
# Batch size
batch_size = 16

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

model = timm.create_model('hf_hub:timm/coatnet_1_rw_224.sw_in1k', pretrained=False)
# Need to change input channels to 1 instead of 3 for greyscale images
# Also change prediction head to 1 dimensional output for binary classification
# Also keep only 1 block in each of the stages
model.stem.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.head.fc = nn.Linear(in_features=768, out_features=1, bias=True)
for i in range(4):
    model.stages[i].blocks = nn.Sequential(model.stages[i].blocks[0])

model.to(device)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Run training
log = {"epoch": [], 
       "train_loss": [], 
       "train_acc": [],
       "val_loss": [],
       "val_acc": []}

for epoch in range(20):
    train_loss = []
    model.train()
    train_correct = 0
    for data in tqdm(train_loader):
        x = data[0].to(device)
        y = data[1].float().to(device)
        pred = model(x).squeeze()
        pred = nn.Sigmoid()(pred)
        # Compute loss
        loss = nn.BCELoss(reduction="mean")(pred, y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward() # backprop
        optimizer.step() # optimize
        # Accuracy
        train_correct += (((pred > 0.5) * y) + ((pred <= 0.5) * (1 - y))).sum().item()
    
    model.eval()
    val_loss = []
    val_correct = 0
    for data in tqdm(val_loader):
        x = data[0].to(device)
        y = data[1].float().to(device)
        pred = model(x).squeeze()
        pred = nn.Sigmoid()(pred)
        # Compute loss
        loss = nn.BCELoss(reduction="mean")(pred, y)
        val_loss.append(loss.item())
        # Accuracy
        val_correct += (((pred > 0.5) * y) + ((pred <= 0.5) * (1 - y))).sum().item()
    
    epoch_train_loss = sum(train_loss) / len(train_loss)
    epoch_train_acc = train_correct / train_size
    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_acc = val_correct / val_size

    with open("training_log.txt", "a") as f:
        line = (str(epoch) + "\t" +
                str(epoch_train_loss) + "\t" +
                str(epoch_train_acc) + "\t" +
                str(epoch_val_loss) + "\t" +
                str(epoch_val_acc) + "\n")
        print(line)
        f.write(line)
    log["epoch"].append(epoch)
    log["train_loss"].append(epoch_train_loss)
    log["train_acc"].append(epoch_train_acc)
    log["val_loss"].append(epoch_val_loss)
    log["val_acc"].append(epoch_val_acc)