# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:18:36 2018

@author: L1YAK01
"""

import numpy as np
import random

np.random.seed(1)
random_nums = np.random.random(5)

Nproducts = 2
varxi= 4 #temporary
np.random.seed(1)
random_mvn = np.random.normal(0, 0.5*varxi, Nproducts)

random.seed(1)
rand = random.gauss([0]*5,np.identity(5))