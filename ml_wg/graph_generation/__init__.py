# ml_wg/graph_generation/__init__.py
from .erdos_renyi import generate_erdos_renyi
from .watts_strogatz import generate_watts_strogatz

__all__ = ['generate_erdos_renyi', 'generate_watts_strogatz']
