import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


use_cuda = torch.cuda.is_available()
use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")

############### Data preprocessing ###################
def create_csv(root='./data/', out_name='labels.csv'):
    subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
    df = pd.DataFrame(columns=['file_path', 'label'])
    for i, path in enumerate(subfolders):
        files = [f.path for f in os.scandir(path) if f.is_file()]
        for f in files:
            df = df.append({'file_path':f, 'label':i}, ignore_index=True)
    df.to_csv(root+out_name, index=False)
    
class ClassificationDataset(Dataset):

    def __init__(self, csv='./data/labels.csv', transform=None):
        self.csv = pd.read_csv(csv)
        self.data_size = len(self.csv)
        self.idx_to_data = dict(zip(range(self.data_size), zip(self.csv['file_path'].tolist(), self.csv['label'].tolist())))
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        filepath, label = self.idx_to_data[int(idx)]
        img = Image.open(filepath)
        if self.transform:
            img, label = self.transform((img, label))
        return img, label

class SquareResize():
    """Adjust aspect ratio of image to make it square"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # assert output_size is int or tuple
        self.output_size = output_size

    def __call__(self, sample):        
        image, label = sample
        h, w = image.size
        if h>w:
            new_w = self.output_size
            scale = new_w/w
            new_h = scale*h
        elif w>h:
            new_h = self.output_size
            scale = new_h/h
            new_w = scale*w
        else:
            new_h, new_w = self.output_size, self.output_size
        new_h, new_w = int(new_h), int(new_w) # account for non-integer computed dimensions (rounds to nearest int)
        image = image.resize((new_h, new_w))
        image = image.crop((0, 0, self.output_size, self.output_size))
        return image, label

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample
        image = np.array(image)/255
        image = image.transpose((2, 0, 1))
        return torch.Tensor(image), label
############### End of data preprocessing ###################

class VGGClassifier(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.features = models.vgg11(pretrained=True).features #512x7x7
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_size),
            torch.nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = F.relu(self.features(x)).reshape(-1, 512*7*7)
        x = self.regressor(x)
        return x

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad=False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad=True

create_csv()

classnames = [f.name for f in os.scandir('./data/') if f.is_dir()]
classname_to_id = dict(zip(classnames, range(len(classnames))))
id_to_classname = dict(zip(classname_to_id.values(), classname_to_id.keys()))
n_classes = len(classnames)

lr = [3e-4, 3e-5] #differential learning rate. #lr[0] is main lr #lr[1] is lr of early layers
weight_decay = 0#1e-4
batch_size = 5
train_split = 0.8
val_split = 0.9

mymodel = VGGClassifier(out_size=n_classes).to(device)

optimizer = torch.optim.Adam([{'params': mymodel.regressor.parameters()},
                                 {'params': mymodel.features.parameters(), 'lr': lr[1]}],
                                 lr=lr[0], weight_decay=weight_decay)
mytransforms = []
mytransforms.append(SquareResize(224))
mytransforms.append(ToTensor())
mytransforms = transforms.Compose(mytransforms)

mydataset = ClassificationDataset(csv='./data/labels.csv', transform=mytransforms)

data_size=len(mydataset)
train_size = int(train_split * data_size)
val_size = int(val_split * data_size) - train_size
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(mydataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_samples = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def train(epochs):
    plt.close()
    mymodel.train()
    
    bcosts = []
    ecosts = []
    valcosts = []
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    #ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(122)
    
    plt.show()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')

    #ax1.set_xlabel('Batch')
    #ax1.set_ylabel('Cost')

    ax2.axis('off')
    img_label_text = ax2.text(0, -5, '', fontsize=15)
    
    for e in range(epochs):
        ecost=0
        valcost=0
        for i, (x, y) in enumerate(train_samples):
            x, y = x.to(device), y.to(device)

            h = mymodel.forward(x) #calculate hypothesis
            cost = F.cross_entropy(h, y, reduction='sum') #calculate cost
            
            optimizer.zero_grad() #zero gradients
            cost.backward() # calculate derivatives of values of filters
            optimizer.step() #update parameters

            bcosts.append(cost.item()/batch_size)
            #ax1.plot(bcosts, 'b', label='Train cost')
            #if e==0 and i==0: ax1.legend()
            
            y_ind=0
            im = np.array(x[y_ind]).transpose(1, 2, 0)
            predicted_class = id_to_classname[h.max(1)[1][y_ind].item()]
            ax2.imshow(im)
            img_label_text.set_text('Predicted class: '+ predicted_class)
            
            fig.canvas.draw()
            ecost+=cost.item()
        #classes_shown=set()
        for i, (x, y) in enumerate(val_samples):
            x, y = x.to(device), y.to(device)

            h = mymodel.forward(x) #calculate hypothesis
            cost = F.cross_entropy(h, y, reduction='sum') #calculate cost

            '''for y_ind, yval in enumerate(y):
                if yval.item() not in classes_shown:
                    classes_shown.add(yval.item())
                    break'''
            y_ind=0
            im = np.array(x[y_ind]).transpose(1, 2, 0)
            predicted_class = id_to_classname[h.max(1)[1][y_ind].item()]
            ax2.imshow(im)
            img_label_text.set_text('Predicted class: '+ predicted_class)
            fig.canvas.draw()
            
            valcost+=cost.item()
        ecost/=train_size
        valcost/=val_size
        ecosts.append(ecost)
        valcosts.append(valcost)
        ax.plot(ecosts, 'b', label='Train cost')
        ax.plot(valcosts, 'r', label='Validation cost')
        if e==0: ax.legend()
        fig.canvas.draw()

        print('Epoch', e, '\tCost', ecost)

def test():
    print('Started evaluation...')
    mymodel.eval() #put model into evaluation mode
    
    #calculate the accuracy of our model over the whole test set in batches
    correct = 0
    for x, y in test_samples:
        x, y = x.to(device), y.to(device)
        h = mymodel.forward(x)
        pred = h.data.max(1)[1]
        correct += pred.eq(y).sum().item()
    return round(correct/len(test_data), 4)

mymodel.freeze()
train(20)
#mymodel.unfreeze()
#train(5)

acc = test()
print('Test accuracy: ', acc)
