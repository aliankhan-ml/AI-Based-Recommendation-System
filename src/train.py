import torch
import clip
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)

class image_title_dataset():
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title

# use your own data
from pandas import *
 
# reading CSV file
data = read_csv("/home/all/Full_data/full_train.csv")
 
# converting column data to list
list_image_path = data['File_path'].tolist()
list_txt = data['Tags'].tolist()
print(len(list_image_path))
print(len(list_txt))

#Load data
dataset = image_title_dataset(list_image_path,list_txt)
print("Tokenization done")
train_dataloader = DataLoader(dataset,batch_size = 5) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)

import matplotlib.pyplot as plt

# Lists to store the loss and accuracy values for each epoch
train_losses = []
train_img_accs = []
train_txt_accs = []

for epoch in range(100):
    correct_img = 0
    correct_txt = 0
    total = 0
    epoch_loss = 0
    
    for batch in train_dataloader :
        optimizer.zero_grad()

        images,texts = batch 

        images= images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        
        _, predicted_img = torch.max(logits_per_image, 1)
        _, predicted_txt = torch.max(logits_per_text, 1)

        correct_img += (predicted_img == ground_truth).sum().item()
        correct_txt += (predicted_txt == ground_truth).sum().item()
        total += len(ground_truth)
        epoch_loss += total_loss.item()

    accuracy_img = (correct_img / total) * 100
    accuracy_txt = (correct_txt / total) * 100
    epoch_loss /= len(train_dataloader)
    train_losses.append(epoch_loss)
    train_img_accs.append(accuracy_img)
    train_txt_accs.append(accuracy_txt)
    print(f'Epoch: {epoch} Training Loss: {total_loss} Train Image Accuracy: {accuracy_img} Train Text Accuracy: {accuracy_txt}')
    
    if epoch % 5 == 4:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'image-accuracy': accuracy_img,
            'text-accuracy': accuracy_txt, 
            }, f"/home/all/Full_model_lr=1*e^-4/model_save_epoch{epoch}.pt".format(epoch+1))
        
        # Plot and save the training loss and accuracy graphs as PNG images
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"/home/all/Full_model_lr=1*e^-4/train_loss_epoch_{epoch}.png")
        plt.clf()
        
        plt.plot(train_img_accs)
        plt.title("Training Image Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.savefig(f"/home/all/Full_model_lr=1*e^-4/train_img_acc_epoch_{epoch}.png")
        plt.clf()
        
        plt.plot(train_txt_accs)
        plt.title("Training Text Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.savefig(f"/home/all/Full_model_lr=1*e^-4/train_txt_acc_epoch_{epoch}.png")
        plt.clf()
    