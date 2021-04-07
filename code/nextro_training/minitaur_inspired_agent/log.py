#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

colors = ['b','r','g']
labels = ['hip0','hip1','hip2']

def makeGraphs(logs):
    logs = pickle.load(open(logs,'rb'))
    x = [x[0] for x in logs['x_y_logs']]
    y = [x[1] for x in logs['x_y_logs']]


    plt.figure(1)
    plt.xlim([-100,100])
    plt.ylim([-100, 100])
    plt.axline((-100,0),(100,0),color='k',ls='--',alpha=0.5)
    plt.axline((0,-10),(0,10),color='k',ls='--',alpha=0.5)
    plt.plot(x,y,c='r')
    plt.xlabel('X-Axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.show()

    for idx in range(0,9,3):
        j = [item[idx] for item in logs['leg_logs']]
        j = np.rad2deg(j)
        plt.figure(2)
        plt.subplot(3,1,idx//3+1)
        length = len(j)
        plt.plot(np.linspace(0, length/30, length), j,c=colors[idx//3],label=labels[idx//3])
        plt.legend(loc='lower left')
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")

    plt.grid(ls='--',c='k',alpha=0.5)
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-log',required=False,type=str,default='logs.p',
                        help="Location of the log file!")
    args = parser.parse_args()
    makeGraphs(args.log)
