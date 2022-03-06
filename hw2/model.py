import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        self.left_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.left_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.left_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.left_conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.left_conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.right_conv1 = nn.Conv2d(384, 128, 3, padding=1)
        self.right_conv2 = nn.Conv2d(192, 64, 3, padding=1)
        self.right_conv3 = nn.Conv2d(96, 32, 3, padding=1)
        self.right_conv4 = nn.Conv2d(48, 16, 3, padding=1)
        self.right_conv5 = nn.Conv2d(16, 6, 1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        x_1 = nn.ReLU()(self.left_conv1(x))
        x_2_a = nn.MaxPool2d(2, 2)(x_1)
        x_2 = nn.ReLU()(self.left_conv2(x_2_a))
        x_3_a = nn.MaxPool2d(2, 2)(x_2)
        x_3 = nn.ReLU()(self.left_conv3(x_3_a))
        x_4_a = nn.MaxPool2d(2, 2)(x_3)
        x_4 = nn.ReLU()(self.left_conv4(x_4_a))
        x_5_a = nn.MaxPool2d(2, 2)(x_4)
        x_5 = nn.ReLU()(self.left_conv5(x_5_a))

        x_6_b = F.interpolate(x_5, scale_factor=(2,2))
        x_6_f = torch.cat((x_4, x_6_b), 1)
        x_6 = nn.ReLU()(self.right_conv1(x_6_f))

        x_7_b = F.interpolate(x_6, scale_factor=(2,2))
        x_7_f = torch.cat((x_3, x_7_b), 1)
        x_7 = nn.ReLU()(self.right_conv2(x_7_f))

        x_8_b = F.interpolate(x_7, scale_factor=(2,2))
        x_8_f = torch.cat((x_2, x_8_b), 1)
        x_8 = nn.ReLU()(self.right_conv3(x_8_f))

        x_9_b = F.interpolate(x_8, scale_factor=(2,2))
        x_9_f = torch.cat((x_1, x_9_b), 1)
        x_9 = nn.ReLU()(self.right_conv4(x_9_f))

        out = self.right_conv5(x_9)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
