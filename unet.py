import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        """first conv layer"""
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        """second conv layer"""
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2=nn.ReLU()

    def forward(self, x: torch.Tensor):
        """apply 2 conv layers and activations"""
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

class DownSample(nn.Module):
    """each step in this contracting path down-samples the feature map with a 2x2 max pooling layer"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)

class UpSample(nn.Module):
    """contrariwise, each step in this expansive path up-samples the feature map with a 2x2 up convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)

class CropAndConcat(nn.Module):
    """
       at every step in the expansive path, the corresponding feature map (FM)
       from the contracting path is concatenated with the current feature map
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2],x.shape[3]]) # corresponding feature map from contracting path
        x = torch.cat([x, contracting_x], dim=1) # the current FM in expansive path
        return x

class UNet(nn.Module):
    """in_channels = number of channels in the input image
    out_channels = number of channels in the result feature map"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in [(in_channels, 64), (64,128), (128,256), (256,512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in [(1024,512), (512,256),(256,128),(128,64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in [(1024,512),(512,256),(256,128),(128,64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64,out_channels, kernel_size=1)

    def forward(self, x:torch.Tensor):
        """x = input image"""
        pass_through = [] # collects outputs of contracing path for later concatenation with the expansive path
        for i in range(len(self.down_conv)): # contracting path
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_conv)): # expansive path
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)
        x = self.final_conv(x)
        return x


