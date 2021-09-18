#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:35:50 2021

@author: liuzhe
"""

import torch
import torch.nn as nn



from single_env_from_csv import *

e = SingleEnv("/Users/liuzhe/github/DPL_study/data/snap.csv", symbol="AP")

x  =3
y = 5


import pandas



y = pandas.DataFrame({'a':[3,4], 'b':[1,2] })
