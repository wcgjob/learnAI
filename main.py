# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Author : 10852
# @Time : 2020/12/24 23:55
# @File : main.pyy
# @Software : PyCharm
# @Description:
"""
import time

import torch


a = torch.randn(50000, 5000)
b = torch.randn(5000, 10000)

t0 = time.time()
c=torch.matmul(a,b)
t1 = time.time()
print(a.device,t1-t0,c.norm())

time.sleep(3)

device = torch.device('cuda')


a = a.to(device)
b = b.to(device)

t0 = time.time()
c=torch.matmul(a,b)
t1 = time.time()

print(a.device,t1-t0,c.norm())

time.sleep(20)

a = a.to(device)
b = b.to(device)

t0 = time.time()
c=torch.matmul(a,b)
t1 = time.time()

print(a.device,t1-t0,c.norm())
print(a.device,t1-t0,c.norm())


