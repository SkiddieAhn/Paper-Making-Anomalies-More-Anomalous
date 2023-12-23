import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze and Excitation block
class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
    

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, mode='train', input_channels1=12, input_channels2=6, input_channels3=84, output_channels=3):
        super(UNet, self).__init__()
        self.mode = mode

        self.inc_x = inconv(input_channels1, 64)
        self.down1_x = down(64, 128)
        self.down2_x = down(128, 256)
        self.down3_x = down(256, 512)

        self.inc_y = inconv(input_channels2, 64)
        self.down1_y = down(64, 128)
        self.down2_y = down(128, 256)
        self.down3_y = down(256, 512)

        self.inc_z = inconv(input_channels3, 64)
        self.down1_z = down(64, 128)
        self.down2_z = down(128, 256)
        self.down3_z = down(256, 512)

        self.transform_xtoy = double_conv(512, 512)
        self.transform_xtoz = double_conv(512, 512)
        self.fusion_conv = double_conv(512*3, 512)
        self.se = SELayer(512)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x, y, z):
        '''
        x: frames (4, 12, 256, 256)
        y: motions (4, 6, 256, 256) 
        z: labels (4, 84, 256, 256)
        '''
        x1 = self.inc_x(x)
        x2 = self.down1_x(x1)
        x3 = self.down2_x(x2)
        x4 = self.down3_x(x3)

        y1 = self.inc_y(y)
        y2 = self.down1_y(y1)
        y3 = self.down2_y(y2)
        y4 = self.down3_y(y3)

        z1 = self.inc_z(z)
        z2 = self.down1_z(z1)
        z3 = self.down2_z(z2)
        z4 = self.down3_z(z3)

        x_y = self.transform_xtoy(x4) # x to y transform (ftom_feature)
        x_z = self.transform_xtoz(x4) # x to z transform (ftol_feature)
        fs = torch.cat((x4, x_y, x_z), dim=1)
        fs = self.fusion_conv(fs)
        fs = self.se(fs)

        x = self.up1(fs, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        # L2 Normalization
        x4 = F.normalize(x4, p=2, dim=1)
        y4 = F.normalize(y4, p=2, dim=1)
        x_y = F.normalize(x_y, p=2, dim=1)
        z4 = F.normalize(z4, p=2, dim=1)
        x_z = F.normalize(x_z, p=2, dim=1)

        return torch.tanh(x), x4, y4, x_y, z4, x_z


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    x1 = torch.ones([4, 12, 256, 256]).cuda()
    x2 = torch.ones([4, 8, 256, 256]).cuda()
    x3 = torch.ones([4, 84, 256, 256]).cuda()
    model = UNet(12, 6, 84, 3).cuda()

    print(model.parameters)
