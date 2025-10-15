import torch
import torch.nn as nn

def main():
    pass

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_layer(out)
        out = self.conv2(out)
        out = self.relu_layer(out)
        return out

class LNR(nn.Module):
    def __init__(self):
        super(LNR, self).__init__()

        self.pool_layer = nn.MaxPool2d(2, 1)

        # Down Convolution block 1
        self.d_conv1 = Conv_Block(3, 64)
        

        # Down Convolution block 2
        self.d_conv2 = Conv_Block(64, 128)

        # Down Convolution block 3
        self.d_conv3 = Conv_Block(128, 256)

        # Down Convolution block 4
        self.d_conv4 = Conv_Block(256, 512)

        # Bottlneck
        self.b_neck = Conv_Block(512, 1024)

        # Up Convolution block 1
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2)
        self.u_conv1 = Conv_Block(1024, 512)

        # Up Convolution block 2
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2)
        self.u_conv2 = Conv_Block(512, 256)

        # Up Convolution block 3
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2)
        self.u_conv3 = Conv_Block(256, 128)

        # Up Convolution block 4
        self.up_conv4 = nn.ConvTranspose2d(128, 128, 2)
        self.u_conv4 = Conv_Block(128, 64)

        # Final 1x1 convolution
        self.out_layer = nn.Conv2d(64, 2, 1)

        # Image switching layers
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        skip_block_1 = self.d_conv1(x)
        out = self.pool_layer(skip_block_1)

        skip_block_2 = self.d_conv1(out)
        out = self.pool_layer(skip_block_2)

        skip_block_3 = self.d_conv1(out)
        out = self.pool_layer(skip_block_3)

        skip_block_4 = self.d_conv1(out)
        out = self.pool_layer(skip_block_4)
        
        out = self.b_neck(out)

        out = self.up_conv1(out)
        out = self.u_conv1(torch.cat(skip_block_4, out, dim=1))       

        out = self.up_conv2(out)
        out = self.u_conv2(torch.cat(skip_block_2, out, dim=1))  

        out = self.up_conv3(out)
        out = self.u_conv3(torch.cat(skip_block_2, out, dim=1))  

        out = self.up_conv4(out)
        out = self.u_conv4(torch.cat(skip_block_1, out, dim=1))  

        self.out_layer(out)

        out = torch.round(self.softmax(out)) 
        out = torch.cat(out[:][0] * x[:][0:3] + out[:][1] * x[:][0:3], out[:][0] * x[:][3:] + out[:][1] * x[:][3:]) # Swap the original images

        return out
    