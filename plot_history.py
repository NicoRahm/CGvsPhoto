# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:44:11 2016

@author: nozick
"""

import matplotlib.pyplot as plt

def plot_history(filename) :
    # load history file
    history = []
    with open('/tmp/history.txt', 'r') as f:
        for line in f:
            history.append(float(line))
        
#    print(history)

    # plot loss data
    plt.close('all')

    plt.plot(history, color='r', label='accuracy')
#    acc_y_max = max(history)
#    plt.ylim(0,acc_y_max*1.1)
    plt.ylim(0,1)    
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    legend = plt.legend(loc='lower right', shadow=False, fontsize='x-large')
    plt.show()


if __name__ == "__main__":    
    plot_history('/tmp/history.npy')