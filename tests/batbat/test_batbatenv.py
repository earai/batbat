import tensorflow as tf
from batbat.batbatenv import BatBatEnv


def test_reset():
    x = BatBatEnv()
    obs = x.reset()
    assert BatBatEnv.PITCHER in obs
    assert BatBatEnv.BATTER in obs
    for k in [BatBatEnv.PITCHER, BatBatEnv.BATTER]:
        y = obs[k]
        assert set(y.keys()) >= {"turn", "n_strikes", "n_balls"}
        assert y["turn"] == 0
        assert y["n_strikes"] == 0
        assert y["n_balls"] == 0


def test_step():
    x = BatBatEnv()
    x.reset()
    pitcher_action_dict = {BatBatEnv.PITCHER: BatBatEnv.FASTBALL}
    pitcher_obs, pitcher_dones, pitcher_rewards, pitcher_info = x.step(pitcher_action_dict)

    batter_action_dict = {BatBatEnv.BATTER: BatBatEnv.NO_SWING}
    batter_obs, batter_dones, batter_rewards, batter_info = x.step(batter_action_dict)


    assert pitcher_dones == {}
    assert pitcher_info == {}
    assert pitcher_rewards == {BatBatEnv.PITCHER: 0,
                               BatBatEnv.BATTER: 0}

    for k in [BatBatEnv.PITCHER, BatBatEnv.BATTER]:

        assert pitcher_obs[k]["turn"] == 1
        assert batter_obs[k]["turn"] == 2
        assert pitcher_obs[k]["n_strikes"] + pitcher_obs[k]["n_balls"] == 0
        assert batter_obs[k]["n_strikes"] + batter_obs[k]["n_balls"] == 1
    assert batter_dones == {}
    assert batter_info == {}
    assert batter_rewards == {BatBatEnv.PITCHER: 0,
                              BatBatEnv.BATTER: 0}
