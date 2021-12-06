import numpy as np
import pandas as pd
import json

def append_to_csv(row, resultPath):
    with open(resultPath, 'a') as fd:
        fd.write(row)

def write_to_csv(row, resultPath):
    with open(resultPath, 'w') as fd:
        fd.write(row)