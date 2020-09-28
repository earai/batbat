from ray.rllib.env.multi_agent_env import MultiAgentEnv
from tensorflow_probability import distributions as tfd
import tensorflow as tf

from batbat.pitch_outcome import hit_distribution, encode_pitch_outcome


class BatBatEnv(MultiAgentEnv):
    """Simplest environment for batting game.

    """
    PITCHER = "pitcher"
    BATTER = "batter"

    FASTBALL = 0
    CURVEBALL = 1

    STRIKE = 0
    BALL = 1
    HIT = 2

    NO_SWING = 0
    SWING = 1

    def __init__(self):
        self.n_strikes = 0
        self.n_balls = 0
        self.pitch_type = None
        self.swing_or_not = None
        self.turn = 0
        self.p_identify_fast_ball = .7
        self.p_identify_curve_ball = .8
        self.p_fast_ball_in_strike_zone = .9
        self.p_curve_ball_in_strike_zone = .5
        self.p_hit_fast_ball = .7
        self.p_hit_curve_ball = .6
        self.p_zone_given_pitch = [self.p_fast_ball_in_strike_zone, self.p_curve_ball_in_strike_zone]
        self.p_hit_given_pitch = [self.p_hit_fast_ball, self.p_hit_curve_ball]

    def reset(self):
        self.n_strikes = 0
        self.n_balls = 0
        self.pitch_type = None
        self.swing_or_not = None
        self.turn = 0

        return {
            self.PITCHER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls},
            self.BATTER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls,
                          "noisy_pitch_type": None}
        }

    def step(self, action_dict):
        if self.turn % 2 == 0:
            return self._step_pitcher(action_dict)
        else:
            return self._step_batter(action_dict)

    def _step_batter(self, action_dict):
        self.swing_or_not = action_dict[self.BATTER]
        self.turn += 1

        joint_dist = hit_distribution(self.pitch_type, self.p_zone_given_pitch, self.p_hit_given_pitch,
                                      self.swing_or_not)

        dones = {}
        x = joint_dist.sample()
        in_zone = x['in_zone']
        hit = x['hit']

        batting_result = encode_pitch_outcome(in_zone.numpy(), self.swing_or_not, hit.numpy())

        if batting_result == self.STRIKE:
            self.n_strikes += 1
        elif batting_result == self.BALL:
            self.n_balls += 1
        elif batting_result == self.HIT:
            dones = {"__all__"}
        else:
            raise ValueError(f"unexpected result: {batting_result}")
        is_strikeout = self.n_strikes >= 3
        is_walk = self.n_balls >= 4
        is_hit = batting_result == self.HIT
        if is_strikeout:
            rewards = {self.PITCHER: 1,
                       self.BATTER: -1}
        elif is_walk:
            rewards = {self.PITCHER: -1,
                       self.BATTER: 1}
        elif is_hit:
            rewards = {self.PITCHER: -2,
                       self.BATTER: 2}
        else:
            rewards = {self.PITCHER: 0,
                       self.BATTER: 0}
        obs = {
            self.PITCHER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls},
            self.BATTER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls,
                          "noisy_pitch_type": None}
        }
        info = {}
        return obs, dones, rewards, info

    def _step_pitcher(self, action_dict):
        self.pitch_type = action_dict[self.PITCHER]
        self.turn += 1
        noisy_pitch_type = tfd.Bernoulli(
            probs=tf.constant([1 - self.p_identify_fast_ball, self.p_identify_curve_ball])[
                self.pitch_type]).sample()
        obs = {
            self.PITCHER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls},
            self.BATTER: {"turn": self.turn, "n_strikes": self.n_strikes, "n_balls": self.n_balls,
                          "noisy_pitch_type": noisy_pitch_type}
        }
        dones = {}
        rewards = {self.PITCHER: 0,
                   self.BATTER: 0}
        info = {}
        return obs, dones, rewards, info
