#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# namelist = ['B1000','B500','B100','C1','C4','hid','E100','E500']
namelist = ['E500']
for name in namelist:
    FNN_without_target_net = np.load('data/FNN_without_target_net_'+name+'_mean_sqs.npy')
    FNN_without_target_net_r = np.load('data/FNN_without_target_net_'+name+'_mean_rewards.npy')
    FNN_with_target_net = np.load('data/FNN_with_target_net_'+name+'_mean_sqs.npy')
    FNN_with_target_net_r = np.load('data/FNN_with_target_net_'+name+'_mean_rewards.npy')

    LSTM_without_target_net = np.load('data/LSTM_without_target_net_'+name+'_mean_sqs.npy')
    LSTM_without_target_net_r = np.load('data/LSTM_without_target_net_'+name+'_mean_rewards.npy')
    LSTM_with_target_net = np.load('data/LSTM_with_target_net_'+name+'_mean_sqs.npy')
    LSTM_with_target_net_r = np.load('data/LSTM_with_target_net_'+name+'_mean_rewards.npy')

    x11 = FNN_without_target_net_r
    x12 = FNN_with_target_net_r
    x21 = FNN_without_target_net
    x22 = FNN_with_target_net

    y11 = LSTM_without_target_net_r
    y12 = LSTM_with_target_net_r
    y21 = LSTM_without_target_net
    y22 = LSTM_with_target_net

    plt.figure(figsize=(20,10))
    plt.subplot(221)
    plt.plot(x11, color='red')
    plt.plot(x12, color='blue')
    plt.ylabel("Rewards")
    plt.legend(['FNN_without_target_net', 'FNN_with_target_net'])
    plt.subplot(223)
    plt.plot(x21, color='red')
    plt.plot(x22, color='blue')
    plt.ylabel("Video Quality")
    plt.xlabel("Video Episode")
    plt.legend(['FNN_without_target_net', 'FNN_with_target_net'])

    plt.subplot(222)
    plt.plot(y11, color='red')
    plt.plot(y12, color='blue')
    plt.ylabel("Rewards")
    plt.legend(['LSTM_without_target_net', 'LSTM_with_target_net'])
    plt.subplot(224)
    plt.plot(y21, color='red')
    plt.plot(y22, color='blue')
    plt.ylabel("Video Quality")
    plt.xlabel("Video Episode")
    plt.legend(['LSTM_without_target_net', 'LSTM_with_target_net'])

    plt.savefig('./fig/all_'+name+'.png')
    # plt.show()
