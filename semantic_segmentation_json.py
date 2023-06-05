# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:22:08 2023

@author: ms024
"""

import json
# simple example
id2label = {0: 'box_area'}
with open('id2label.json', 'w') as fp:
    json.dump(id2label, fp)