import os
import pandas as pd
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict


# run = 5
path = "results/ubuntu/V_34_blue/F_all_no_nan/result.csv"
data = pd.read_csv(path, header=None)
print(data)

g = 250
F_list = [ g/8*i for i in range(9)] 
F_list[0] = 1
M_list = F_list


x = [ i for i in range(0,40000,100)]
x.append(39999)

# x = [ i for i in range(0,20000,100)]
# x.append(19999)

# x = [ i for i in range(0,10000,100)]
# x.append(9999)

# x = [ i for i in range(0,3000,30)]
# x.append(2999)

tmp = x


# run time = 50
graph_list = [ [ [None]*50 for _ in range(9)] for _ in range(9) ]
graph_list = np.array(graph_list)
for x, F in enumerate(F_list):
    for y, M in enumerate(M_list):

        # df = data.loc[(data[3]==F) & (data[5]==M) & (data[11]==39999) & (data[6]==g)]
        # print("F {} M {} run {}".format(F, M, df.shape[0]))

        print("x {} y {}".format(x, y))
        for i in tmp:
            df = data.loc[(data[3]==F) & (data[5]==M) & (data[11]==i) & (data[6]==g)]
            for index, row in df.iterrows():
                run_idx = row[0]
                if row[8] == "NaN":
                    graph_list[x, y, run_idx] = None
                    continue
                if graph_list[x, y, run_idx] != None:
                    dic = graph_list[x, y, run_idx]
                    dic["y"].append(row[8])
                if graph_list[x, y, run_idx] == None:
                    dic = defaultdict()
                    dic["run_idx"] = run_idx
                    dic["y"] = [row[8]]
                    graph_list[x, y, run_idx] = dic
# a = input()
print("graph data collected")
print("ploting")


x = [ i for i in range(0,40000,100)]
x.append(39999)

# x = [ i for i in range(0,20000,100)]
# x.append(19999)

# x = [ i for i in range(0,10000,100)]
# x.append(9999)

# x = [ i for i in range(0,3000,30)]
# x.append(2999)

plt.figure()
count = 1
for x_idx, F in enumerate(F_list):
    for y_idx, M in enumerate(M_list):
        for dic in graph_list[x_idx, y_idx]:
            if dic != None :
                y = dic["y"]
                ax = plt.subplot(9, 9, count)
                title = "F = " + str(F) + " M = " + str(M)
                ax.set_title(title)
                plt.plot(x, y)
            if dic == None:
                print("F {} M {}".format(F, M))
        count += 1
plt.suptitle("AC g = 250 step = 40000")
plt.show()



