import torch
from torch import nn
from dataset import corpus
from gensim.models import KeyedVectors
from torch.utils.data import random_split,DataLoader
from model import fcm
from torch.autograd.variable import Variable


if __name__ == '__main__':
    wv=KeyedVectors.load_word2vec_format('wv1.txt', binary=False)
    with open('corpus.txt','r',encoding='utf-8')as f:
        data=[line for line in f.readlines()]
    
    data=corpus(data[:1000],wv)
    train_s,test_s=random_split(data,[900,100])
    train=DataLoader(train_s,batch_size=100)
    test=DataLoader(test_s,batch_size=100)
    
    epochs=1000
    lr=3e-3
    min_loss=7e-2
    max_ac=0.9
    
    fcm=fcm().cuda()
    criterion=nn.BCELoss()
    optimizer=torch.optim.AdamW(fcm.parameters(),lr)  #采用随机梯度下降的方法
    loss_data=[]
    ac=[]
    for _ in range(epochs):
        sum_loss=0
        train_correct=0
        for data in train:
            x,y=data
            x,y=Variable(torch.tensor(x)).float().cuda(),Variable(torch.tensor(y)).cuda()
            
            out=fcm(x).float()
            y=y.unsqueeze(1).float()
            loss=criterion(out,y)
            p_loss=loss.data
            
            mask=out.ge(0.5).float()  #以0为阈值进行分类
            correct=(mask==y).sum()  #计算正确预测的样本个数
            acc=correct.item()/x.size(0)  #计算精度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_correct+=acc
            sum_loss+=p_loss
            
        ave_loss=sum_loss/len(train)
        ave_c=train_correct/len(train)
        
        if (_+1)%50==0:
            loss_data.append(ave_loss.cpu().float())
            ac.append(ave_c)
            print('*'*10)
            print('epoch {}'.format(_+1))  #误差
            print('loss is {:.4f}'.format(ave_loss))
            print('acc is {:.4f}'.format(ave_c))  #精度  
            if(ave_c>max_ac):
                max_ac=ave_c
                torch.save(fcm,f'ac_{ave_c}_loss_{ave_loss}.pt')
            if(ave_loss<min_loss):
                break
    torch.save(fcm,f'ac_{ave_c}_loss_{ave_loss}.pt')
    print('^'*20)
   
    # fcm=torch.load('ac_0.97_loss_0.10155420750379562.pt').cuda()
    sum_loss=0
    test_correct=0
    fcm.eval()
    for data in test:
        x,y=data
        x,y=Variable(torch.tensor(x)).float().cuda(),Variable(torch.tensor(y)).cuda()
        
        out=fcm(x)
        y=y.unsqueeze(1).float()
        loss=criterion(out,y)
        p_loss=loss.data
        
        mask=out.ge(0.5).float()  #以0为阈值进行分类
        correct=(mask==y).sum()  #计算正确预测的样本个数
        acc=correct.item()/x.size(0)  #计算精度
        test_correct+=acc
        sum_loss+=p_loss
    print('loss is {:.4f}'.format(sum_loss/len(test)))
    print('acc is {:.4f}'.format(test_correct/len(test)))  #精度