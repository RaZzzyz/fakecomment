from torch import nn, relu
import torch

class fcm(nn.Module):
    def __init__(self) :
        super(fcm,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bilstm=nn.LSTM(input_size=12,hidden_size=4, \
                        num_layers=2, \
                        bidirectional=True)

        self.h0 = torch.randn(4,8,4).cuda()
        self.c0= torch.randn(4,8,4).cuda()
        self.dense=nn.Linear(8,1)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        x=x.view(len(x),16,8)
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        input=torch.cat((x1,x2,x3,x4),dim=2)
        out,(hn,cn)=self.bilstm (input,(self.h0,self.c0))
        x=self.dense(out[:,-1,:]).squeeze(0)
        x=self.sig(x)
        return x
        