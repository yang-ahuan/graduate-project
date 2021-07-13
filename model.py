import torch
import torch.nn as nn
import torch.nn.functional as F

class MixCNN(nn.Module):
    def __init__(self):
        super(MixCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=10),
            nn.MaxPool1d(kernel_size=4, stride=2),
            # nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10),
            nn.MaxPool1d(kernel_size=4, stride=2),
            # nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )
        self.fully_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*491, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        # print(out.size())
        n, c, l = out.size()
        out = out.view(n, c*l)
        out = self.fully_layer(out)
        return out

class DeepSEA(nn.Module):
    def __init__(self):
        super(DeepSEA, self).__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=320, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fully_layer = nn.Sequential(
            nn.Linear(960*115, 925),
            nn.ReLU(),
            nn.Linear(925, 6),
        )

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        # print(out.size())
        n, c, l = out.size()
        out = out.view(n, c*l)
        out = self.fully_layer(out)
        return out

class SepCNN(nn.Module):
    def __init__(self):
        super(SepCNN, self).__init__()
        self.dna_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10),
            nn.MaxPool1d(kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10),
            nn.MaxPool1d(kernel_size=4, stride=1),
            nn.ReLU()
        )
        self.meth_conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.fully_layer = nn.Sequential(
            nn.Linear(190304, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
            # nn.Sigmoid()
        )

    def forward(self, x):
        batch, row, col = x.size()
        dna, meth = x[:,:4,:], torch.reshape(x[:,4,:], (batch, 1, -1))
        dna = self.dna_conv_layer(dna)
        meth = self.meth_conv_layer(meth)

        batch, row, col = dna.size()
        dna = dna.view(batch, row*col)
        batch, row, col = meth.size()
        meth = meth.view(batch, row*col)

        out = torch.cat((dna, meth), dim=1)
        # print(out.size())
        out = self.fully_layer(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad, batch_norm, bias):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, pad, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channel) if batch_norm else None
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm: x = self.batch_norm(x)
        x = self.activation(x)
        return x

class InputModule(nn.Module):
    def __init__(self, in_channel, config, batch_norm, bias): # int, config(out_channel, kernel_size, stride, pad), bool, bool
        super(InputModule,self).__init__()
        self.branch_0 = ConvBlock(in_channel, *config[0], batch_norm, bias)
        self.branch_1 = ConvBlock(in_channel, *config[1], batch_norm, bias)
        self.branch_2 = ConvBlock(in_channel, *config[2], batch_norm, bias)
    def forward(self, x):
        out_0 = self.branch_0(x)
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        # print(out_0.size(),out_1.size(),out_2.size())
        return torch.cat([out_0, out_1, out_2], dim=1)

class InceptionModule(nn.Module):
    def __init__(self, in_channel, config, batch_norm, bias):
        super(InceptionModule, self).__init__()
        self.branch_0 = ConvBlock(in_channel, *config[0], batch_norm, bias)
        self.branch_1 = nn.Sequential(
            ConvBlock(in_channel, *config[1][0], batch_norm, bias),
            ConvBlock(config[1][0][0], *config[1][1], batch_norm, bias)
        )
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channel, *config[2][0], batch_norm, bias),
            ConvBlock(config[2][0][0], *config[2][1], batch_norm, bias)
        )
        self.branch_3 = nn.Sequential(
            nn.MaxPool1d(*config[3][0]),
            ConvBlock(in_channel, *config[3][1], batch_norm, bias),
        )
    def forward(self, x):
        out_0 = self.branch_0(x)
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        return torch.cat([out_0, out_1, out_2, out_3], dim=1)

class TestNet(nn.Module):
    def __init__(self, batch_norm=True, bias=False):
        super(TestNet, self).__init__() # config: out_channel, kernel_size, stride, padding
        # Input layer (1,3,7)
        self.input_conv = InputModule(in_channel=4,
                                      config=[[16,1,1,0],[32,3,1,1],[64,7,1,3]],
                                      batch_norm=batch_norm, bias=bias)
        # self.input_pool = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        # Integrate layer (1,3)
        self.integrate_conv = nn.Sequential(
            ConvBlock(in_channel=112, out_channel=112, kernel_size=1, stride=1, pad=0,
                      batch_norm=batch_norm, bias=bias),
            ConvBlock(in_channel=112, out_channel=128, kernel_size=3, stride=1, pad=1,
                      batch_norm=batch_norm, bias=bias)
        )
        # self.integrate_pool = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        # Inception layer a (1,3,5,3)
        self.incept_layer_a = InceptionModule(in_channel=128,
                                              config=[[32,1,1,0],
                                                      [[56,1,1,0],[64,3,1,1]],
                                                      [[56,1,1,0],[64,5,1,2]],
                                                      [[3,1,1],[32,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        # Inception layer b (1,3,5,3) (1,7,9,7)
        self.incept_layer_b0 = InceptionModule(in_channel=192,
                                              config=[[48,1,1,0],
                                                      [[96,1,1,0],[96,3,1,1]],
                                                      [[96,1,1,0],[96,5,1,2]],
                                                      [[3,1,1],[48,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        self.incept_layer_b1 = InceptionModule(in_channel=288,
                                              config=[[64,1,1,0],
                                                      [[128,1,1,0],[128,7,1,3]],
                                                      [[128,1,1,0],[128,9,1,4]],
                                                      [[7,1,3],[64,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        # Inception layer c (1,7,9,7)
        self.incept_layer_c = InceptionModule(in_channel=384,
                                              config=[[64,1,1,0],
                                                      [[192,1,1,0],[192,7,1,3]],
                                                      [[192,1,1,0],[192,9,1,4]],
                                                      [[7,1,3],[64,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        self.incept_last_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Linear(100,6)
        )

    def forward(self, x):
        # Input layer
        x = self.input_conv(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Integrate layer
        x = self.integrate_conv(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer a
        x = self.incept_layer_a(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer b
        x = self.incept_layer_b0(x)
        x = self.incept_layer_b1(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer c
        x = self.incept_layer_c(x)
        x = self.incept_last_pool(x)
        # classifier
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layer(x)
        return x

class LeNupLikeNet(nn.Module):
    def __init__(self, batch_norm=True, bias=False):
        super(LeNupLikeNet, self).__init__() # config: out_channel, kernel_size, stride, padding
        # Input layer (2,3,5)
        self.input_conv = InputModule(in_channel=5,
                                      config=[[16,2,2,0],[32,3,2,1],[64,5,2,2]],
                                      batch_norm=batch_norm, bias=bias)
        # self.input_pool = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        # Inception layer a (1,3,5,3)
        self.incept_layer_a = InceptionModule(in_channel=112,
                                              config=[[32,1,1,0],
                                                      [[56,1,1,0],[64,3,1,1]],
                                                      [[56,1,1,0],[64,5,1,2]],
                                                      [[3,1,1],[32,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        # Inception layer b (1,3,5,3) (1,7,9,7)
        self.incept_layer_b0 = InceptionModule(in_channel=192,
                                              config=[[48,1,1,0],
                                                      [[96,1,1,0],[96,3,1,1]],
                                                      [[96,1,1,0],[96,5,1,2]],
                                                      [[3,1,1],[48,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        self.incept_layer_b1 = InceptionModule(in_channel=288,
                                              config=[[64,1,1,0],
                                                      [[128,1,1,0],[128,7,1,3]],
                                                      [[128,1,1,0],[128,9,1,4]],
                                                      [[7,1,3],[64,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        # Inception layer c (1,7,9,7)
        self.incept_layer_c = InceptionModule(in_channel=384,
                                              config=[[64,1,1,0],
                                                      [[192,1,1,0],[192,7,1,3]],
                                                      [[192,1,1,0],[192,9,1,4]],
                                                      [[7,1,3],[64,1,1,0]]],
                                              batch_norm=batch_norm, bias=bias)
        # Pooling layer
        # self.incept_pool = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        self.incept_last_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layer
        self.fc_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Linear(100,6)
        )

    def forward(self, x):
        # Input layer
        x = self.input_conv(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer a
        x = self.incept_layer_a(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer b
        x = self.incept_layer_b0(x)
        x = self.incept_layer_b1(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, ceil_mode=True)
        # Incept layer c
        x = self.incept_layer_c(x)
        x = self.incept_last_pool(x)
        # classifier
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layer(x)
        return x

class CnnConcatLstm(nn.Module):
    def __init__(self, batch_norm=False, bias=True):
        super(CnnConcatLstm, self).__init__()
        # Input layer (2,3,5)
        self.input_conv = InputModule(in_channel=5,
                                      config=[[16,2,2,0],[32,3,2,1],[64,5,2,2]],
                                      batch_norm=batch_norm, bias=bias)
        # self.input_pool = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        self.rnn = nn.GRU(input_size=16+32+64, hidden_size=256, num_layers=2, batch_first=True)
        self.fc_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Linear(100,6)
        )

    def forward(self, x):
        x = self.input_conv(x)
        # x = self.input_pool(x)
        # (batch, feat, seq_len) --> (batch, seq_len, feat)
        x = torch.transpose(x, 1, 2)
        # Extract hidden state
        _, x = self.rnn(x)
        x = torch.cat((x[0],x[1]), dim=1)
        x = F.relu(x)
        x = self.fc_layer(x)
        return x

