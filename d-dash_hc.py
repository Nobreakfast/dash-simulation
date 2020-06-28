#!/usr/bin/env python3

import copy
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import torch
from dataclasses import dataclass
import math

# global variables
# - DQL
CH_HISTORY = 2                  # number of channel capacity history samples
BATCH_SIZE = 1000
EPS_START = 0.8
EPS_END = 0.0
LEARNING_RATE = 1e-4
# - FFN
N_I = 3 + CH_HISTORY            # input dimension (= state dimension)
N_H1 = 128
N_H2 = 256
N_O = 4
# - D-DASH
BETA = 2
GAMMA = 50
DELTA = 0.001
B_MAX = 20
B_THR = 10
T = 2  # segment duration
TARGET_UPDATE = 20
LAMBDA = 0.9

is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

device = torch.device("cpu")


@dataclass
class State:
    """
    $s_t = (q_{t-1}, F_{t-1}(q_{t-1}), B_t, \bm{C}_t)$, which is a modified
    version of the state defined in [1].
    """

    sg_quality: int
    sg_size: float
    buffer: float
    ch_history: np.ndarray

    def tensor(self):
        return torch.tensor(
            np.concatenate(
                (
                    np.array([
                        self.sg_quality,
                        self.sg_size,
                        self.buffer]),
                    self.ch_history
                ),
                axis=None
            ),
            dtype=torch.float32
        )


@dataclass
class Experience:
    """$e_t = (s_t, q_t, r_t, s_{t+1})$ in [1]"""

    state: State
    action: int
    reward: float
    next_state: State


class ReplayMemory(object):
    """Replay memory based on a circular buffer (with overlapping)"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * self.capacity
        self.position = 0
        self.num_elements = 0

    def push(self, experience):
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        if self.num_elements < self.capacity:
            self.num_elements += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_num_elements(self):
        return self.num_elements


class ActionSelector(object):
    """
    Select an action based on the exploration policy.
    """

    def __init__(self, num_actions, num_segments, greedy_policy=False):
        self.steps_done = 0
        self.num_actions = num_actions
        self.num_segments = num_segments
        self.greedy_policy = greedy_policy

    def reset(self):
        self.steps_done = 0

    def increse_step_number(self):
        self.steps_done += 1

    def action(self, state, policy_net):
        if self.greedy_policy:
            with torch.no_grad():
                return int(torch.argmax(policy_net(state.tensor().to(device))))
        else:
            sample = random.random()
            x = 20 * (self.steps_done / self.num_segments) - 6.  # scaled s.t. -6 < x < 14
            eps_threshold = EPS_END + (EPS_START - EPS_END) / (1. + math.exp(x))
            # self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return int(torch.argmax(policy_net(state.tensor().to(device))))
            else:
                return random.randrange(self.num_actions)

# policy-network based on FNN with 2 hidden layers
class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fn = torch.nn.Sequential(
            torch.nn.Linear(N_I, N_H1),
            torch.nn.ReLU(),
            torch.nn.Linear(N_H1, N_H2),
            torch.nn.ReLU(),
            torch.nn.Linear(N_H2, N_O),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fn(x)
        return x

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size = N_I,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True
        )
        self.out = torch.nn.Linear(128, 4)
    def forward(self, x):
        x = x.view(-1, 1, 5)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out

class DQN(object):
    def __init__(self, NET, TARGET, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.NET = NET
        self.TARGET = TARGET
        if NET == 'FNN':
            self.policy_net = FNN()
        elif NET == 'LSTM':
            self.policy_net = RNN()
        if TARGET:
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(self.BATCH_SIZE)

    def simulate_dash(self, sss, bws, phase, epoch):
        # initialize parameters
        num_segments = sss.shape[0]  # number of segments
        num_qualities = sss.shape[1]  # number of quality levels


        if phase == 'train':
            # initialize action_selector
            selector = ActionSelector(num_qualities, num_segments, greedy_policy=False)
        elif phase == 'test':
            selector = ActionSelector(num_qualities, num_segments, greedy_policy=True)
        else:
            sys.exit(phase+" is not supported.")


        ##########
        # training
        ##########
        num_episodes = epoch
        mean_sqs = np.empty(num_episodes)  # mean segment qualities
        mean_rewards = np.empty(num_episodes)  # mean rewards
        for i_episode in range(num_episodes):
            sqs = np.empty(num_segments-CH_HISTORY)
            rewards = np.empty(num_segments-CH_HISTORY)

            # initialize the state
            sg_quality = random.randrange(num_qualities)  # random action
            state = State(
                sg_quality=sg_quality,
                sg_size=sss[CH_HISTORY-1, sg_quality],
                buffer=T,
                ch_history=bws[0:CH_HISTORY]
            )
            for t in range(CH_HISTORY, num_segments):
                sg_quality = selector.action(state, self.policy_net)
                sqs[t-CH_HISTORY] = sg_quality

                # update the state
                tau = sss[t, sg_quality] / bws[t]
                buffer_next = T + max(0, state.buffer-tau)
                next_state = State(
                    sg_quality=sg_quality,
                    sg_size=sss[t, sg_quality],
                    buffer=buffer_next,
                    ch_history=bws[t-CH_HISTORY+1:t+1]
                )

                # calculate reward (i.e., (4) in [1]).
                downloading_time = next_state.sg_size/next_state.ch_history[-1]
                rebuffering = max(0, downloading_time-state.buffer)
                rewards[t-CH_HISTORY] = next_state.sg_quality \
                    - BETA*abs(next_state.sg_quality-state.sg_quality) \
                    - GAMMA*rebuffering - DELTA*max(0, B_THR-next_state.buffer)**2

                # store the experience in the replay memory
                experience = Experience(
                    state=state,
                    action=sg_quality,
                    reward=rewards[t-CH_HISTORY],
                    next_state=next_state
                )
                # print(experience)
                self.memory.push(experience)

                # move to the next state
                state = next_state

                #############################
                # optimize the policy network
                #############################
                if self.memory.get_num_elements() < self.BATCH_SIZE:
                    continue
                experiences = self.memory.sample(self.BATCH_SIZE)
                state_batch = torch.stack([experiences[i].state.tensor().to(device)
                                            for i in range(self.BATCH_SIZE)])
                next_state_batch = torch.stack([experiences[i].next_state.tensor().to(device)
                                                for i in range(self.BATCH_SIZE)])
                action_batch = torch.tensor([experiences[i].action
                                                for i in range(self.BATCH_SIZE)], dtype=torch.long).to(device)
                reward_batch = torch.tensor([experiences[i].reward
                                                for i in range(self.BATCH_SIZE)], dtype=torch.float32).to(device)

                # $Q(s_t, q_t|\bm{w}_t)$ in (13) in [1]
                # 1. policy_net generates a batch of Q(...) for all q values.
                # 2. columns of actions taken are selected using 'action_batch'.
                state_action_values = self.policy_net(state_batch).gather(1, action_batch.view(self.BATCH_SIZE, -1))

                # $\max_{q}\hat{Q}(s_{t+1},q|\bar{\bm{w}}_t$ in (13) in [1]
                # TASK 2: Replace policy_net with target_net.
                if self.TARGET:
                    next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
                else:
                    next_state_values = self.policy_net(next_state_batch).max(1)[0].detach()


                # expected Q values
                expected_state_action_values = reward_batch + (LAMBDA * next_state_values)

                # loss fuction, i.e., (14) in [1]
                mse_loss = torch.nn.MSELoss(reduction='mean')
                loss = mse_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                # TASK2: Implement target network
                # # update the target network
                if self.TARGET:
                    if t % TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

            # processing after each episode
            selector.increse_step_number()
            mean_sqs[i_episode] = sqs.mean()
            mean_rewards[i_episode] = rewards.mean()
            print("Mean Segment Quality[{0:2d}]: {1:E}".format(i_episode, mean_sqs[i_episode]))
            print("Mean Reward[{0:2d}]: {1:E}".format(i_episode, mean_rewards[i_episode]))

        return (mean_sqs, mean_rewards)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_video_trace",
        help="training video trace file name; default is 'bigbuckbunny.npy'",
        default='bigbuckbunny.npy',
        type=str)
    parser.add_argument(
        "--test_video_trace",
        help="testing video trace file name; default is 'bear.npy'",
        default='bear.npy',
        type=str)
    parser.add_argument(
        "-C",
        "--channel_bandwidths",
        help="channel bandwidths file name; default is 'bandwidths.npy'",
        default='bandwidths.npy',
        type=str)
    parser.add_argument(
        "-N",
        "--neural_network_type",
        help="change the trainning module: default is 'FNN",
        default='FNN',
        type=str
    )
    parser.add_argument(
        "-R",
        "--episode",
        help="change the episode time: default is 100",
        default=100,
        type=int
    )
    parser.add_argument(
        "-T",
        "--target_net",
        help="with or without target net; default is False",
        default=False,
        type=bool
    )
    parser.add_argument(
        "-F",
        "--file_name",
        help="indicate the filename",
        default="FNN_withou_target_net",
        type=str
    )
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch_size for memory; default is 1000",
        default=1000,
        type=int
    )
    parser.add_argument(
        "-H",
        "--channel_history",
        help="channel history; default is 2",
        default=2,
        type=int
    )
    parser.add_argument(
        "-L",
        "--hidden_layer",
        help="hidden layer number; default is 128",
        default=128,
        type=int
    )

    args = parser.parse_args()
    train_video_trace = args.train_video_trace
    test_video_trace = args.test_video_trace
    channel_bandwidths = args.channel_bandwidths
    net_type = args.neural_network_type
    episode = args.episode
    target = args.target_net
    filename = args.file_name
    batchsize = args.batch_size
    channel_history = args.channel_history
    hidden_layer = args.hidden_layer
   
    # initialize channel BWs and replay memory
    bws = np.load(channel_bandwidths)  # channel bandwdiths [bit/s]
    dqn = DQN(net_type, target, batchsize)

    # training phase
    sss = np.load(train_video_trace)        # segment sizes [bit]
    train_mean_sqs, train_mean_rewards = dqn.simulate_dash(sss, bws, 'train', episode)

    # save the model
    torch.save(dqn.policy_net, './model/'+filename+'_policy.pkl')
    if target:
        torch.save(dqn.target_net, './model/'+filename+'_target.pkl')

    # testing phase
    sss = np.load(test_video_trace)        # segment sizes [bit]
    test_mean_sqs, test_mean_rewards = dqn.simulate_dash(sss, bws, 'test', episode)

    # concatenate the train and test data
    mean_sqs = np.concatenate((train_mean_sqs, test_mean_sqs), axis=None)
    mean_rewards = np.concatenate((train_mean_rewards, test_mean_rewards), axis=None)

    # save data
    np.save('./data/'+filename+'_mean_sqs', mean_sqs)
    np.save('./data/'+filename+'_mean_rewards', mean_rewards)
    # plot results
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(mean_rewards)
    axs[0].set_ylabel("Reward")
    axs[0].vlines(len(train_mean_rewards), *axs[0].get_ylim(), colors='red', linestyles='dotted')
    axs[1].plot(mean_sqs)
    axs[1].set_ylabel("Video Quality")
    axs[1].set_xlabel("Video Episode")
    axs[1].vlines(len(train_mean_rewards), *axs[1].get_ylim(), colors='red', linestyles='dotted')
    plt.savefig('./fig/d-dash_'+filename+'.png')
    plt.show()
    #  input("Press ENTER to continue...")
    plt.close('all')
