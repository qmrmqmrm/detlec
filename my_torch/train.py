
import torch
from my_torch.loader.a2d2_loader import A2D2_Loader
from my_torch.modeling.rcnn import GeneralizedRCNN
from torch.utils.data import DataLoader
from torch.autograd import Variable

PIXEL_MEAN = [0.0,0.0,0.0]
PIXEL_STD = [1.0,1.0,1.0]


model = GeneralizedRCNN(PIXEL_MEAN,PIXEL_STD)

path = "/media/dolphin/intHDD/birdnet_data/my_a2d2"
data_load = A2D2_Loader(path)
print("\ndata_load.len : ",data_load.len)
train_loader = DataLoader(dataset=data_load,
                          batch_size=2,
                          shuffle=True,
                          num_workers=2)


criterion = torch.nn.BCELoss(size_average = True)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
# Training loop
train_loader_iter = iter(train_loader)
print(train_loader_iter)
batch = next(train_loader_iter)
print(batch)
for epoch in range(2):
    for i, data in enumerate(batch):
        # get the inputs
        print(data)
        # wrap them in Variable
        data= Variable(data)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data)

        # Compute and print loss
        loss = criterion(y_pred, data)
        print(epoch, i, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
