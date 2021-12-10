# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:09:59 2021

@author: Khoir Salam
"""

import gzip, os, sys
import numpy as np
from scipy.stats import multivariate_normal
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import sys
from tkinter import *

#Bagian 1
def download(filename, source='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)


def load_fashion_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data

def load_fashion_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

#Bagian 2
## Load the training set
train_data = load_fashion_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_fashion_mnist_labels('train-labels-idx1-ubyte.gz')
## Load the testing set
test_data = load_fashion_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_fashion_mnist_labels('t10k-labels-idx1-ubyte.gz')
print(train_data.shape)
# (60000, 784) ## 60k 28x28 images
print(test_data.shape)
# (10000, 784) ## 10k 2bx28 images
print(np.max(train_data))

products = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(len(products))
## Define a function that displays an image given its vector representation
def show_image(x, label):
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.title(products[label], size=15)
    plt.axis('off')

plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10, 10, i+1)
    show_image(test_data[i,:], test_labels[i])
plt.tight_layout()
plt.show()

# normalize
X_train = np.zeros(train_data.shape)
for i in range(train_data.shape[0]):
    X_train[i,:] = train_data[i,:] / np.max(train_data[i,:])
X_test = np.zeros(test_data.shape)
for i in range(test_data.shape[0]):
    X_test[i,:] = test_data[i,:] / np.max(test_data[i,:])
    
    
#Bagian 3    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n*n, 512)
        self.fc21 = nn.Linear(512, 32) # mu        # must change to (512, 2) if you want a 2D VAE 
        self.fc22 = nn.Linear(512, 32) # sigma     # must change to (512, 2) if you want a 2D VAE 
        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, n*n)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, n*n))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
#Bagian 4
    
torch.manual_seed(1)

cuda = torch.cuda.is_available()
batch_size = 512 #128
log_interval = 20
epochs = 20
n = 28

device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = DataLoader(np.reshape(X_train, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(np.reshape(X_test, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#Bagian 5
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, n*n), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

#Bagian 6
def train(epoch):
    model.train()
    batch_idx = 0
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        batch_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

#Bagian 7    
def test(epoch):
    model.eval()
    losses = []
    i = 0
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                N = min(data.size(0), 8)
                comparison = torch.cat([data[:N],
                                      recon_batch.view(batch_size, 1, n, n)[:N]])
                save_image(comparison.cpu(),
                         'images/rekonstruksi/reconstruction_' + str(epoch) + '.png', nrow=N)
                i += 1

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#Bagian 8
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 32).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, n, n),
                   'images/sampel/sample_' + str(epoch) + '.png')
torch.save(model, 'models/vae.pth')

with torch.no_grad():
    mu, _ = model.encode(torch.from_numpy(X_test).float().to(device))
mu = mu.cpu().numpy()
  
plt.figure(figsize=(12, 10)) 
plt.scatter(mu[:, 0], mu[:, 1], c=test_labels, cmap='jet'), plt.colorbar()
plt.xlabel({i:products[i] for i in range(len(products))}, fontsize=15)
plt.show()


#================================GUI=========================================#
def button1():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\sampel\sample_17.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button2():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\sampel\sample_18.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button3():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\sampel\sample_19.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button4():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\sampel\sample_20.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button5():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\rekonstruksi\reconstruction_17.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button6():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\rekonstruksi\reconstruction_18.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button7():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\rekonstruksi\reconstruction_19.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button8():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\rekonstruksi\reconstruction_20.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button9():
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = r'E:\Tugas\Kuliah\MATKUL\Semester 7\Pengcit\UTS Pengolahan Citra\Python-Image-Processing-Cookbook-master\Chapter 09\images\fashion_mnist_test.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

mGui = Tk()
mGui.geometry("250x170")
l = Label(mGui, text = "A variational autoencoder to reconstruct and generate images")
l.config(font =("Times New Rowman", 14))
l.pack()
button9 = Button(mGui,text ='Gambar Referensi ',command = button9, height=5, width=20).pack()
button1 = Button(mGui,text ='Sample 17',command = button1, height=5, width=20).pack()
button2 = Button(mGui,text ='Sample 18',command = button2, height=5, width=20).pack()
button3 = Button(mGui,text ='Sample 19',command = button3, height=5, width=20).pack()
button4 = Button(mGui,text ='Sample 20',command = button4, height=5, width=20).pack()
button5 = Button(mGui,text ='Rekonstruksi 17',command = button5, height=5, width=20).pack()
button6 = Button(mGui,text ='Rekonstruksi 18',command = button6, height=5, width=20).pack()
button7 = Button(mGui,text ='Rekonstruksi 19',command = button7, height=5, width=20).pack()
button8 = Button(mGui,text ='Rekonstruksi 20',command = button8, height=5, width=20).pack()


mGui.mainloop()

