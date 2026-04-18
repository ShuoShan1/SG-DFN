import logging
import torch
from datetime import datetime
import torch
import pickle
import numpy as np
import time
from typing import List


def set_device(device_id):
    if device_id == -1:
        device = 'cpu'
        logging.info("\n no gpu found, program is running on cpu! \n")
        return device
    else:
        device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            logging.info("\n no gpu found, program is running on cpu! \n")
        return device
   

def load_entityid(file_path):
    entityid = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            index = line.split()[0]
            entityid.append(index)
    entityid = list(map(int, entityid))
    return entityid

