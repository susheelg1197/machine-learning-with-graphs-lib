import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from collections import deque
import random

class GraphEnvironment:
    """
    A simple graph-based environment for reinforcement learning.
    """
    def __init__(self, graph, start_node, goal_node):
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.current_node = start_node

    def reset(self):
        self.current_node = self.start_node
        return self.current_node

    def step(self, action):
        next_node = list(self.graph.neighbors(self.current_node))[action]
        self.current_node = next_node
        reward = 1 if next_node == self.goal_node else -0.1
        done = next_node == self.goal_node
        return next_node, reward, done

class GraphPolicyNetwork(nn.Module):
    """
    A simple policy network that operates on graph-structured data.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphPolicyNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.softmax(x, dim=1)
    
class DQNAgent:
    def __init__(self, network, learning_rate=0.01, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)

    def act(self, state, action_space_size):
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_probs = self.network(*state.get_graph_data())
                return action_probs[state.current_node].max(0)[1].item()
        else:
            return random.choice(range(action_space_size))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * self.network(*next_state.get_graph_data())[next_state.current_node].max(0)[1].item()
            target_f = self.network(*state.get_graph_data())
            target_f[state.current_node][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.network(*state.get_graph_data()), target_f)
            loss.backward()
            self.optimizer.step()

def train_policy_network(agent, environment, episodes=100, batch_size=32):
    for episode in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            action = agent.act(state, len(list(environment.graph.neighbors(state.current_node))))
            next_state, reward, done = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay