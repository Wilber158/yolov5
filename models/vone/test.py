import pickle
import numpy as np
import pandas as pd
import xlsxwriter
import torch
from __init__ import get_model
'''
data = pd.DataFrame(pickle.load(open('/home/andrewc/Output/results.pkl', "rb")))
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
data.to_excel(writer, sheet_name='results', index=False)
writer.save()

'''
model = get_model(pretrained=False)
model.load_state_dict(pickle.load(open('/home/andrewc/Output/latest_checkpoint.pth.tar', "rb")))
model.eval()
