import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from dqn import DQN

GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000
BATCH_SIZE = 32
EXPLORATION_MAX = 5e-2
EXPLORATION_MIN = 1e-5
EXPLORATION_DECAY = 0.99


class Agent:
    def __init__(self, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.Q_policy = DQN(action_space)
        self.Q_target = copy.deepcopy(self.Q_policy)
        if torch.cuda.is_available():
            self.Q_policy = self.Q_policy.cuda()
            self.Q_target = self.Q_target.cuda()
        self.optimizer = optim.Adam(self.Q_policy.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        q_values = self.Q_policy(state)[0]

        # best action
        if np.random.rand() < self.exploration_rate:
            max_q_index = [random.randrange(self.action_space)]
        else:
            max_q_index = [torch.max(q_values, 0).indices.cpu().numpy().tolist()]

        max_q_one_hot = one_hot_embedding(max_q_index, self.action_space)
        return max_q_index, max_q_one_hot, q_values

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.cat(tuple(action for action in action_batch))
        reward_batch = torch.cat(tuple(reward for reward in reward_batch))
        reward_batch = reward_batch.view(len(reward_batch), 1)
        next_state_batch = torch.cat(tuple(next_state for next_state in next_state_batch))
        current_prediction_batch = self.Q_policy(state_batch)
        next_prediction_batch = self.Q_target(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + GAMMA * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        self.optimizer.zero_grad()
        loss = self.loss_func(q_value, y_batch.detach())
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]