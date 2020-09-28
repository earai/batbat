import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy

from batbat.batbatenv import BatBatEnv

"""
function(pitch_type, prob_zone(pitch_type), swing ,p_zone(pitchtype))
    -> give joint dist of whether is in zone, whether there was a swing, was there a hit and outcome
"""


def hit_distribution(pitch_type, p_zone_given_pitch, p_hit_given_pitch, swing):
    """ Joint distribution for whether we hit the pitch"""
    return tfd.JointDistributionNamed(dict(
        in_zone=tfd.Bernoulli(probs=tf.cast(p_zone_given_pitch[pitch_type], tf.float64)),
        hit=lambda in_zone: tfd.Bernoulli(probs=tf.cast(p_hit_given_pitch[pitch_type] * in_zone * swing, tf.float64))
    ))


def pitch_outcome(in_zone, swing, hit):
    if (in_zone and not hit) or (not in_zone and swing):
        return BatBatEnv.STRIKE
    elif not in_zone and not swing:
        return BatBatEnv.BALL
    elif hit:
        return BatBatEnv.HIT
    else:
        raise ValueError(f"unexpected outcome {in_zone}, {swing}, {hit}")
