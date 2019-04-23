import pickle as pkl
import numpy as np

TOTAL_TRACE = 10;

def parse_trace_file(i):
    in_path = "../raw_data/"
    trace_name = "trace%d.log"
    cur_file=in_path+trace_name%(i);
    f=open(cur_file, "r");
    count = 0
    arrival_time = []
    for j in range(TOTAL_TRACE):
        arrival_time.append([]);
    for s in f:
        count+=1;
        #if count>200000: break;#test purpose
        if count%5000000==0: print(count);
        try:
            temp = s.split(" ")
            from_id = int(temp[0]);
            beat_num = int(temp[1]);
            receive_t = int(temp[3]);
            arrival_time[from_id].append([beat_num, receive_t]);
        except:
            f.close();
            return arrival_time, count
    f.close();
    return arrival_time, count

def get_directed(i, j):
    arrival_time, _ = parse_trace_file(i)
    return arrival_time[j], len(arrival_time[j])


if __name__ == "__main__":
    in_path = "../raw_data/"
    out_path = "../pkl_data/"
    trace_name = "trace%d.log"
    pkl_name = "trace%d.pkl"
    for i in range(TOTAL_TRACE):
        if (i==1):
            cur_file=in_path+trace_name%(i);
            print(cur_file)
            arrival_time, c = parse_trace_file(cur_file)
            print(c)
            #cur_file=out_path+pkl_name%(i);
            #print(cur_file)
            #dump_pkl(arrival_time, cur_file);
