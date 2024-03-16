# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:17:16 2022

@author: javie
"""
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.get_device_name())