import unittest
import networkx as nx
from ml_wg.reinforcement_learning.graph_rl import GraphEnvironment, GraphPolicyNetwork, DQNAgent

class TestGraphRL(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()
        self.start_node = 0
        self.goal_node = 33
        self.environment = GraphEnvironment(self.graph, self.start_node, self.goal_node)
        self.network = GraphPolicyNetwork(in_channels=1, hidden_channels=16, out_channels=2)
        self.agent = DQNAgent(self.network)

    def test_environment_reset(self):
        state = self.environment.reset()
        self.assertEqual(state, self.start_node)

    def test_environment_step(self):
        self.environment.reset()
        next_state, reward, done = self.environment.step(0)
        self.assertIsNotNone(next_state)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(done)

    def test_agent_act(self):
        state = self.environment.reset()
        action = self.agent.act(state, len(list(self.graph.neighbors(state))))
        self.assertIsInstance(action, int)

if __name__ == "__main__":
    unittest.main()
