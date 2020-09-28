import tensorflow as tf
from batbat.pitch_outcome import hit_distribution, encode_pitch_outcome
import pytest


@pytest.mark.parametrize('pitch_type, swing, expected_in_zone, expected_hit',
                         [(0, 0, 0, 0),
                          (0, 1, 0, 0),
                          (1, 0, 1, 0),
                          (1, 1, 1, 1)])
def test_hit_distribution(pitch_type, swing, expected_in_zone, expected_hit):
    p_zone_given_pitch = [0, 1]
    p_hit_given_pitch = [0, 1]
    x_dist = hit_distribution(pitch_type, p_zone_given_pitch, p_hit_given_pitch, swing)
    x = x_dist.sample()
    assert x['in_zone'] == expected_in_zone
    assert x['hit'] == expected_hit


@pytest.mark.parametrize('in_zone, swing, hit, expected_outcome',
                         [(0, 0, 0, 1),
                          (1, 0, 0, 0),
                          (1, 1, 0, 0),
                          (1, 1, 1, 2),
                          (0, 1, 0, 0)])
def test_encode_pitch_outcome(in_zone, swing, hit, expected_outcome):
    x = encode_pitch_outcom(in_zone, swing, hit)
    assert x == expected_outcome
