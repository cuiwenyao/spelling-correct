from numpy import argmin
import torch
import pynvml

def get_min_used_gpu_id(metric="memory"):
    if not torch.cuda.is_available():
        print("You do even not have a CUDA! Please go to sleep!!!!!!!!!")
        return 0
    gpu_num=torch.cuda.device_count()
    if gpu_num==0:
        print("You do even not have a GPU! Please go to sleep!!!!!!!!!")
        return 0

    pynvml.nvmlInit()
    info_list=[]
    for i in range(gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        if metric=="memory":
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info_list.append(info.used)
        elif metric == "power":
            info = pynvml.nvmlDeviceGetPowerUsage(handle)
            info_list.append(info)
    # print(info_list)
    # print(argmin(info_list))
    return argmin(info_list)

if __name__=="__main__":
    get_min_used_gpu_id("memory")
    get_min_used_gpu_id("power")