import ast
import numpy as np 
import pandas as pd

def extract_classes_size(log_file):
    classes_labels = None
    count_list = []
    with open(log_file,'r') as log:
        for line in log:
            if line.startswith("count of classes"):
                arr_1, arr_2 = line.split('array(')[1:3]  
                rest_of_arr_2 = log.readline().split(')')[0]
                arr_2 += rest_of_arr_2
                
                if classes_labels is None:
                    classes_labels = np.array(ast.literal_eval(arr_1.strip('), ')))
                count_list.append(np.array(ast.literal_eval(arr_2)))
    log.close()
    MAP = {
    0: "ACTUATOR",
    1: "BOX",
    2: "CABLE",
    3: "FLOOR",
    4: "GAUGE",
    5: "PIPESUPPORT",
    6: "PIPE",
    7: "STRUCTURE",
    8: "VALVE",
    255: "UNLABLED(AUGMENT)"
                        }
    idx_to_names = [ MAP[x] for x in classes_labels ]
    df = pd.DataFrame(count_list,columns=idx_to_names)
    df.to_csv("./results/classes_size.csv",header=True,index=False)

extract_classes_size("/home/ubuntu/code/minkowski/outputs/FacilityArea5Dataset/9_classes/2020-06-24_13-04/2020-06-24_13-04.txt")