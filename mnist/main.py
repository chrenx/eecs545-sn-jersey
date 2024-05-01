import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import timeit
import unittest
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter



def track_loss(tb_writer, loss, epoch, data_loader_len, batch_idx, mode):
    step_num = epoch * data_loader_len + batch_idx

    tb_writer.add_scalar(
        f"{mode} loss", 
        loss,
        step_num
    )



def adjust_learning_rate(optimizer, iter, each, printout):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = 0.001 * (0.95 ** (iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    printout.set_description("Learning rate = ",lr)
    return lr




def train(model, device, train_loader, optimizer, epoch, printout, tb_writer):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # send the image, target to the device
        data, target = data.to(device), target.to(device)
        # flush out the gradients stored in optimizer
        optimizer.zero_grad()
        # pass the image to the model and assign the output to variable named output
        output = model(data)

        # calculate the loss (use nll_loss in pytorch)
        loss = F.nll_loss(output, target)
     
    
        data_loader_len = len(train_loader)
        track_loss(tb_writer, loss, epoch, data_loader_len, batch_idx, 'train')

        # do a backward pass
        loss.backward()
        # update the weights
        optimizer.step()

        
        
        if batch_idx % 100 == 0:
            printout.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # losses_1.append(loss.item())
            # losses_2.append(100. * batch_idx / len(train_loader))


def test(model, device, test_loader, epoch, printout, tb_writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
          
            # send the image, target to the device
            data, target = data.to(device), target.to(device)
            # pass the image to the model and assign the output to variable named output
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum').item()
            test_loss += loss # sum up batch loss
          
            data_loader_len = len(test_loader)
            track_loss(tb_writer, loss, epoch, data_loader_len, batch_idx, 'test')

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    

    printout.set_description('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    acc = 100. * correct / len(test_loader.dataset)
    return acc
    # avg_loss.append(test_loss)
    # accuracy.append(100. * correct / len(test_loader.dataset))


def main():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using ", device)


    from MnistDataset import MnistDataset
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.ToTensor(),

        transforms.Normalize((0.3150,), (0.2218,)) 
        # mean: 0.3150
        # std: 0.2218
    ])
    double_mnist_train_data = MnistDataset('custom_datasets/train', data_transforms)
    double_mnist_val_data = MnistDataset('custom_datasets/val', data_transforms)



    train_dataloader = Data.DataLoader(dataset=double_mnist_train_data, batch_size=128, shuffle=True)
    test_dataloader = Data.DataLoader(dataset=double_mnist_val_data, batch_size=128, shuffle=True) 
    print("dataloader done ...")


    model = Net().to(device)


    ## Define Adam Optimiser with a learning rate of 0.01
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    start = timeit.default_timer()
    print("start training ...")
    tb_writer = SummaryWriter("./tb/")
    with tqdm(range(0, 100)) as max_epochs:
        highest_acc = 1
        printout = max_epochs
        for epoch in max_epochs:
            
            lr = adjust_learning_rate(optimizer, epoch, 1.616, printout)
            # learning_rate.append(lr)
            train(model, device, train_dataloader, optimizer, epoch, printout, tb_writer)

            test_acc = 0
            if epoch % 2 == 0:
                test_acc = test(model, device, test_dataloader, epoch, printout, tb_writer)
                print(f"TEST acc at epoch {epoch}: {test_acc}")

            if test_acc > highest_acc and (test_acc-highest_acc) / highest_acc > 0.01 and test_acc > 60:
                highest_acc = test_acc
                torch.save(
                    {
                    'model_state_dict': model.state_dict()
                    }, 
                    f"ckpt/best_double_mnist_{epoch}.pth"
                )
            
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 
                    f"ckpt/double_mnist_{epoch}.pth"
                )
    stop = timeit.default_timer()
    print('Total time taken: {} seconds'.format(int(stop - start)))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define a conv layer with output channels as 16, kernel size of 3 and stride of 1
        self.conv11 = nn.Conv2d(1, 16, 3, 1) # Input = 1x64x64  Output = 16x62x62
        self.conv12 = nn.Conv2d(1, 16, 5, 1) # Input = 1x64x64  Output = 16x60x60
        self.conv13 = nn.Conv2d(1, 16, 7, 1) # Input = 1x64x64  Output = 16x58x58
        self.conv14 = nn.Conv2d(1, 16, 9, 1) # Input = 1x64x64  Output = 16x56x56

        # define a conv layer with output channels as 32, kernel size of 3 and stride of 1
        self.conv21 = nn.Conv2d(16, 32, 3, 1) # Input = 16x62x62 Output = 32x60x60
        self.conv22 = nn.Conv2d(16, 32, 5, 1) # Input = 16x60x60 Output = 32x56x56
        self.conv23 = nn.Conv2d(16, 32, 7, 1) # Input = 16x58x58 Output = 32x52x52
        self.conv24 = nn.Conv2d(16, 32, 9, 1) # Input = 16x56x56  Output = 32x48x48

        # define a conv layer with output channels as 64, kernel size of 3 and stride of 1
        self.conv31 = nn.Conv2d(32, 64, 3, 1) # Input = 32x60x60 Output = 64x58x58
        self.conv32 = nn.Conv2d(32, 64, 5, 1) # Input = 32x56x56 Output = 64x52x52
        self.conv33 = nn.Conv2d(32, 64, 7, 1) # Input = 32x52x52 Output = 64x46x46
        self.conv34 = nn.Conv2d(32, 64, 9, 1) # Input = 32x48x48 Output = 64x40x40


        # define a max pooling layer with kernel size 2
        self.maxpool = nn.MaxPool2d(2) # Output = 64x11x11
        #self.maxpool1 = nn.MaxPool2d(1)
        # define dropout layer with a probability of 0.25
        self.dropout1 = nn.Dropout(0.25)
        # define dropout layer with a probability of 0.5
        self.dropout2 = nn.Dropout(0.5)

        # define a linear(dense) layer with 128 output features
        self.fc11 = nn.Linear(64*29*29, 256)
        self.fc12 = nn.Linear(64*26*26, 256)      # after maxpooling 2x2
        self.fc13 = nn.Linear(64*23*23, 256)
        self.fc14 = nn.Linear(64*20*20, 256)

        # define a linear(dense) layer with output features corresponding to the number of classes in the dataset
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(256, 128)

        self.fc33 = nn.Linear(128*4,99)
        #self.fc33 = nn.Linear(64*3,10)


    def forward(self, inp):
        # Use the layers defined above in a sequential way (folow the same as the layer definitions above) and 
        # write the forward pass, after each of conv1, conv2, conv3 and fc1 use a relu activation. 


        x = F.relu(self.conv11(inp))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        #print(x.shape)
        #x = torch.flatten(x, 1)
        x = x.view(-1,64*29*29)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(inp))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        #x = torch.flatten(x, 1)
        y = y.view(-1,64*26*26)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(inp))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        #x = torch.flatten(x, 1)
        z = z.view(-1,64*23*23)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        ze = F.relu(self.conv14(inp))
        ze = F.relu(self.conv24(ze))
        ze = F.relu(self.maxpool(self.conv34(ze)))
        #x = torch.flatten(x, 1)
        ze = ze.view(-1,64*20*20)
        ze = self.dropout1(ze)
        ze = F.relu(self.fc14(ze))
        ze = self.dropout2(ze)
        ze = self.fc24(ze)

        out_f = torch.cat((x, y, z, ze), dim=1)
        #out_f1 = torch.cat((out_f, ze), dim=1)
        out = self.fc33(out_f)

        output = F.log_softmax(out, dim=1)
        return output


if __name__ == '__main__':
    main()

    