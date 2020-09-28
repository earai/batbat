
from gym.spaces import Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class BatBatEnv(MultiAgentEnv):
    """Simplest environment for batting game."""

    def __init__(self, config):
        self.sheldon_cooper = config.get("sheldon_cooper", False)
        self.action_space = Discrete(5 if self.sheldon_cooper else 3)
        self.observation_space = Discrete(5 if self.sheldon_cooper else 3)
        self.player1 = "player1"
        self.player2 = "player2"
        self.last_move = None
        self.num_moves = 0

        # For test-case inspections (compare both players' scores).
        self.player1_score = self.player2_score = 0

    def reset(self):
        self.last_move = (0, 0)
        self.num_moves = 0
        return {
            self.player1: self.last_move[1],
            self.player2: self.last_move[0],
        }

    def step(self, action_dict):
        pass


