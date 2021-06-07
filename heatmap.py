import os
import pandas as pd
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#[run_number, pop_size, pop_update_rule, F, cost, M, group_size, mutation_rate, cooperators[round], defectors[round], avReward, round]

def merge_all(path):
    inpath =  path + "*.csv"
    outpath = path + "result.csv"
    csv_list = glob.glob(inpath)
    print(len(csv_list))
    for i in csv_list:
        fr = open(i,"rb").read()
        with open(outpath,"ab") as f:
            f.write(fr)
    print("down")

def read_data(path):
    data = pd.read_csv(path, header=None)
    return data

def get_R(data, g, timestep):
    df = data.loc[data[11] == timestep]
    print(df)

    # g = 250
    F_list = [ g/8*i for i in range(9)] 
    F_list[0] = 1
    M_list = F_list

    R = np.empty(shape=(9,9))
    for x, F in enumerate(F_list):
        for y, M in enumerate(M_list):
            sum_c = 0
            run = 0
            df2 = df.loc[(df[3]==F) & (df[5]==M) & (df[6]==g)]
            for index, row in df2.iterrows():
                if not np.isnan(row[8]):
                    run += 1
                    sum_c += row[8]
            R[8-y, x] = sum_c / run
    print(R)

    return R

def plot_heatmap(path_q, path_ac):
    q_data = read_data(path_q)
    ac_data = read_data(path_ac)
    q_R = get_R(q_data, 100, 19999)
    ac_R = get_R(ac_data, 250, 39999)

    q_R = np.round(q_R, 2)
    ac_R = np.round(ac_R, 2)

    f, (ax1,ax2,ax3) = plt.subplots(figsize=(20,5), ncols=3)

    sns.heatmap(q_R, ax=ax1, xticklabels=[1,"n/8","2n/8","3n/8","4n/8","5n/8","6n/8","7n/8","n"], yticklabels=["n","7n/8","6n/8","5n/8","4n/8","3n/8","2n/8","n/8",1], annot=True)
    ax1.set_title("R1 AC g = 100")
    ax1.set_xlabel("F")
    ax1.set_ylabel("M")

    sns.heatmap(ac_R, ax=ax2, xticklabels=[1,"n/8","2n/8","3n/8","4n/8","5n/8","6n/8","7n/8","n"], yticklabels=["n","7n/8","6n/8","5n/8","4n/8","3n/8","2n/8","n/8",1], annot=True)
    ax2.set_title("R2 AC g = 250")
    ax2.set_xlabel("F")
    ax2.set_ylabel("M")

    # R3 ------------------------------
    R3 = ac_R - q_R

    R3 = np.round(R3, 2)

    sns.heatmap(R3, ax=ax3, xticklabels=[1,"n/8","2n/8","3n/8","4n/8","5n/8","6n/8","7n/8","n"], yticklabels=["n","7n/8","6n/8","5n/8","4n/8","3n/8","2n/8","n/8",1], annot=True)
    ax3.set_title("R2 - R1")
    ax3.set_xlabel("F")
    ax3.set_ylabel("M")
    # ---------------------------------

    f.suptitle("g100 run50 step20000 | g250 run50 step40000")
    plt.show()

if __name__ == "__main__":
    # merge_all("results/ubuntu/V_34_blue/F_all_no_nan/")

    #注意 除 的次数 run = ?

    path_ac100 = "results/ubuntu/V_34/result.csv"
    path_ac250 = "results/ubuntu/V_34_blue/F_all_no_nan/result.csv"
    # path_ac = path_q
    plot_heatmap(path_ac100, path_ac250)