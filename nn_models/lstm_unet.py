import torch.nn as nn
import torch.nn.functional as F
import torch
from nn_models.convlstm import ConvLSTM

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)                 #b,c,h,w
        diffX = x1.size()[2] - x2.size()[2]                # making sure res dimension is the same
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, use_LSTM=True, parallel_encoder = False, factor = 0, lstm_layers = 2):
        super(UNet_ConvLSTM, self).__init__()
        self.use_LSTM = use_LSTM
        self.parallel_encoder = parallel_encoder
        self.factor = factor
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512,128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        if use_LSTM:
            self.convlstm = ConvLSTM(input_dim=512,
                                     hidden_dim=[512]*lstm_layers,
                                     kernel_size=(3,3),
                                     num_layers=lstm_layers,
                                     batch_first=False,
                                     bias=True,
                                     return_all_layers=False)

    def forward(self, x, prev_hidden=None):
        #b, t, c, h, w = x.shape
        if self.use_LSTM:
            if self.parallel_encoder == True:
                x1 = self.inc(x.view(t*b, c, h, w))
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                _, c, h, w = x5.shape
                data = x5.view(b, t, c, h, w)
            else:
                b, t, c, h, w = x.shape
                if self.factor == 0:
                    factor = t
                else:
                    factor = self.factor
                x = x.view(int((t*b)/factor), factor, c, h, w)  # does nothing?
                x = torch.unbind(x, dim=1)                      # removed dim= n and get n matrices
                data = []
                
                for item in x:  # item.shape =  b,c,h,w
                    x1 = self.inc(item)
                    x2 = self.down1(x1)
                    x3 = self.down2(x2)
                    x4 = self.down3(x3)
                    x5 = self.down4(x4)
                    data.append(x5.unsqueeze(0))   # b,c,h,w
                _, c, h, w = x5.shape
                data = torch.cat(data, dim=0)
                data = data.view(t, b, c, h, w)
            lstm, _ = self.convlstm(data, prev_hidden)
            test = lstm[0][ :,-1, :, :, :]  # test size? = b,c,h,w
        else:
            b, c, h, w = x.shape
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            test = x5
        x = self.up1(test, x4[:b, :, :, :])
        x = self.up2(x, x3[:b, :, :, :])
        x = self.up3(x, x2[:b, :, :, :])
        x = self.up4(x, x1[:b, :, :, :])
        x = self.outc(x)
        return x, test