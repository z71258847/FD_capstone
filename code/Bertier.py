
# coding: utf-8

# In[4]:

from trace_extract import parse_trace_file
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle as pkl

# In[5]:

class Jacobson():
    def __init__(self, gamma, beta, phi):
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.cur_delay = 0
        self.cur_var = 0
        self.safety_margin = 0
    
    def compute_safety_margin(self, actual_arr, expect_arr):
        err = actual_arr - expect_arr - self.safety_margin
        #print("error is ",err);
        new_delay = self.cur_delay + self.gamma*err
        #print("new_delay is ",new_delay);
        new_var = self.cur_var + self.gamma*(err-self.cur_var)
        #print("new_var is ",new_var);
        self.cur_delay = new_delay
        self.cur_var = new_var
        self.safety_margin = self.beta*new_delay+self.phi*new_var
        #print("safety_margin is ",self.safety_margin);
        return self.safety_margin


# In[6]:

class Bertier_monitor():
    def __init__(self, self_id, monitoring_id, history_size):
        self.ita=100000000;
        self.id = self_id
        self.monitoring = monitoring_id
        self.history_size = history_size
        self.arrival_history = deque(maxlen=history_size)
        self.expected_arrival = 0
        self.safety_margin = 0
        self.jacob = Jacobson(0.125, 1, 4);
        self.seq_num = -1;
        self.suspect_intervals=[]
        self.U = 0;
    
    def forward(self, seq_num, arrival_time):
        if (self.seq_num == -1):
            self.seq_num=seq_num
            self.U=arrival_time
            self.arrival_history.append(arrival_time)
            self.expected_arrival = self.U + self.ita;
            self.safety_margin = self.jacob.compute_safety_margin(arrival_time, arrival_time)
        elif (self.seq_num+1==seq_num):
            self.seq_num=seq_num
            if (self.expected_arrival + self.safety_margin < arrival_time):
                self.suspect_intervals.append([self.expected_arrival + self.safety_margin, arrival_time])
            self.arrival_history.append(arrival_time)
            self.safety_margin = self.jacob.compute_safety_margin(arrival_time, self.expected_arrival)
            self.cal_expectation()
        elif (self.seq_num<seq_num):
            '''doesn't work
            self.suspect_intervals.append([self.expected_arrival + self.safety_margin, arrival_time])
            expected_interval = self.expected_arrival - self.arrival_history[-1] + self.safety_margin
            while (self.seq_num<seq_num-1):
                self.seq_num+=1;
                attempt_arrival_time = self.arrival_history[-1] + expected_interval
                self.arrival_history.append(attempt_arrival_time)
                #self.safety_margin = self.jacob.compute_safety_margin(attempt_arrival_time, self.expected_arrival)
                #self.cal_expectation()
            self.arrival_history.append(arrival_time)
            self.safety_margin = self.jacob.compute_safety_margin(arrival_time, self.expected_arrival)
            self.cal_expectation()
            '''
            self.seq_num=seq_num
            self.arrival_history.clear()
            self.jacob = Jacobson(0.125, 1, 4);
            self.U=arrival_time
            self.arrival_history.append(arrival_time)
            self.expected_arrival = self.U + self.ita;
            self.safety_margin = self.jacob.compute_safety_margin(arrival_time, arrival_time)
    
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


# In[ ]:

in_path = "../raw_data/"
trace_name = "trace%d.log"
cur_file=in_path+trace_name%(1);
arrival_times, c = parse_trace_file(cur_file)
print(c)


for j in range(10):
    if (j==1): continue;
    print("monitoring process %d"%(j))
    print("total receive %d"%(len(arrival_times[j])))
    monitor = Bertier_monitor(1, j, 10)
    for (k,trace) in enumerate(arrival_times[j]):
        n=trace[0]
        t=trace[1]
        expt=monitor.expected_arrival/1e9
        sft=monitor.safety_margin/1e9
        maxt=expt+sft
        if (k%1000000==1):
            print("expected and safety margin:", expt, sft)
            print("maximum waiting time", maxt)
            print("actual time", t/1e9)
        monitor.forward(n, t)
    print(len(monitor.suspect_intervals))
    f=open("suspect%d_%d.pkl"%(1,j), "wb");
    pkl.dump(monitor.suspect_intervals, f)
    pkl.dump(arrival_times[j][0], f)
    pkl.dump(arrival_times[j][-1], f)
    f.close()
    

# In[ ]:



