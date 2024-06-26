import torch.nn as nn 
import torch.nn.functional as F
import torch

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

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