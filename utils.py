import os
import numpy as np
import torch
import time

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_best_cindex():
    path = './file'
    c_index = []
    list_name = os.listdir(path)
    for name in list_name:
        c_index.append(float(name.split('-')[2]))
    max_cindex = np.max(c_index)

    load_path = ''
    for name in list_name:
        if str(max_cindex) in name:
            load_path = name

    file_name = 'best_c_index-{:.4f}.pth'.format(max_cindex)
    return os.path.join(path,load_path,file_name)

def save_model(model,optimizer,epoch,file_name):
    saved_path = os.path.join('./model_saved', time.strftime('%Y_%m_%d', time.localtime(time.time())))
    create_path(saved_path)
    torch.save({
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch
    },os.path.join(saved_path,file_name))

def record_best_model(model,file_name,archive_name):
    saved_path = os.path.join('./',archive_name,time.strftime('%Y_%m_%d',time.localtime(time.time())))
    create_path(saved_path)
    path = os.path.join(saved_path,file_name)
    torch.save(model.state_dict(),path)
