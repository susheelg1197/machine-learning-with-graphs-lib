import networkx as nx
from ml_wg.reinforcement_learning.graph_rl import GraphEnvironment, GraphPolicyNetwork, DQNAgent, train_policy_network

# Create a graph and an environment
G = nx.karate_club_graph()
environment = GraphEnvironment(G, start_node=0, goal_node=33)

# Initialize the policy network
network = GraphPolicyNetwork(in_channels=1, hidden_channels=16, out_channels=G.number_of_nodes())
agent = DQNAgent(network)

# Train the policy network
train_policy_network(agent, environment, episodes=100, batch_size=32)

# Testing the trained model
state = environment.reset()
done = False
while not done:
    action = agent.act(state, len(list(G.neighbors(state))))
    state, reward, done = environment.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")
