#!/usr/bin/env python
# coding: utf-8

# In[1]:


from trace_extract import parse_trace_file
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device(1 if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
import time


# In[43]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__();
        self.conv1 = nn.Conv1d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv1d(8, 1, 4)
        self.pooling = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):#16
        x=F.relu(self.conv1(x));
        x=self.dropout(x);
        x=self.pooling(x);#8
        x=F.relu(self.conv2(x));
        x=self.dropout(x);
        x=self.pooling(x);#4
        x=self.conv3(x)
        return x.view(-1);
        


# In[44]:


def asymetric_loss(output, target, alpha=0.75):
    if (output<target):
        loss = 2.0*alpha*((output - target)**2)
    else:
        loss = 2.0*(1-alpha)*((output - target)**2)
    return loss

# In[57]:


class CNN_monitor():
    def __init__(self, self_id, monitoring_id, history_size=16):
        self.ita=100000000;
        self.id = self_id
        self.monitoring = monitoring_id
        self.history_size = history_size
        self.arrival_history = deque(maxlen=history_size)
        self.margin_history = np.zeros(16);
        self.expected_arrival = 0
        self.safety_margin = Variable(torch.tensor([0])).to(device)
        self.cnn_module = SimpleCNN().to(device);
        self.seq_num = -1;
        self.suspect_intervals=[]
        self.U = 0;
        self.lr=1e-2
        self.decay=5e-6
        self.optimizer=optim.SGD(self.cnn_module.parameters(),
                             lr=self.lr,
                             weight_decay=self.decay)
        self.criterion = asymetric_loss#nn.MSELoss()
        self.loss = 0;
    
    def forward(self, seq_num, arrival_time):
        if (self.seq_num == -1):
            self.seq_num=seq_num
            self.U=arrival_time
            self.arrival_history.append(arrival_time)
            self.expected_arrival = self.U + self.ita;
            x=Variable(torch.tensor(self.margin_history).view(1, 1, 16).float()).to(device)
            self.safety_margin = self.cnn_module.forward(x)
        elif (self.seq_num+1==seq_num):
            start_time=time.time();
            self.seq_num=seq_num
            if (self.expected_arrival + self.safety_margin.item()*self.ita < arrival_time):
                self.suspect_intervals.append([self.expected_arrival + self.safety_margin, arrival_time])
            self.arrival_history.append(arrival_time)
            #print("append arrival:", time.time()-start_time);start_time=time.time();
            optimal_margin = (arrival_time-self.expected_arrival)/self.ita
            self.margin_history=np.roll(self.margin_history, -1);
            self.margin_history[-1]=optimal_margin
            optimal_margin = Variable(torch.tensor([optimal_margin]).float()).to(device)
            #print("compute optimal:", time.time()-start_time);start_time=time.time();
            self.optimizer.zero_grad()
            self.loss= self.criterion(self.safety_margin, optimal_margin)
            #print("compute loss:", time.time()-start_time);start_time=time.time();
            self.loss.backward()
            self.optimizer.step()
            #print("back prop:", time.time()-start_time);start_time=time.time();
            
            x=Variable(torch.tensor(self.margin_history).view(1, 1, 16).float()).to(device)
            self.safety_margin = self.cnn_module.forward(x)
            #print("estimate next:", time.time()-start_time);start_time=time.time();
            self.cal_expectation()
            
        elif (self.seq_num<seq_num):
            self.seq_num=seq_num
            self.arrival_history.clear();
            self.U=arrival_time
            self.arrival_history.append(arrival_time)
            self.expected_arrival = self.U + self.ita;
    
    def cal_expectation(self):
        if (len(self.arrival_history)<self.history_size):
            k = len(self.arrival_history);
            temp_U = self.arrival_history[-1]/(k)+(k-1)*self.U/(k)
            self.U = temp_U
            self.expected_arrival = self.U + (k+1)/2*self.ita;
        else:
            temp = (self.arrival_history[-1]-self.arrival_history[0])/(self.history_size-1)
            self.expected_arrival += temp
        return self.expected_arrival


# In[50]:


in_path = "../raw_data/"
trace_name = "trace%d.log"
cur_file=in_path+trace_name%(1);
arrival_times, c = parse_trace_file(cur_file)
print(c)


# In[58]:

for j in range(10):
    if (j==1): continue;
    print("monitoring process %d"%(j))
    print("total receive %d"%(len(arrival_times[j])))
    monitor = CNN_monitor(1, j, 10)
    computation_times=[]
    error_history=[]
    for (k,trace) in enumerate(arrival_times[j]):
        n=trace[0]
        t=trace[1]
        expt=monitor.expected_arrival/1e9
        sft=monitor.safety_margin.item()
        maxt=expt+sft
        error_history.append(t/1e9-maxt)
        if (k%1000000==1):
            print("expected and safety margin:", expt, sft)
            print("maximum waiting time", maxt)
            print("actual time", t/1e9)
        start_time=time.time();
        monitor.forward(n, t)
        computation_times.append(time.time()-start_time);
    print(len(monitor.suspect_intervals))
    f=open("CNN_suspect%d_%d.pkl"%(1,j), "wb");
    pkl.dump(monitor.suspect_intervals, f)
    pkl.dump(arrival_times[j][0], f)
    pkl.dump(arrival_times[j][-1], f)
    pkl.dump(error_history, f)
    pkl.dump(computation_times, f)
    f.close()
