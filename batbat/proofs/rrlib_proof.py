from ray import tune
import gym

import tensorflow as tf
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy import build_tf_policy


def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch["actions"]) * train_batch["rewards"])


def main():
    #tune.run(PPOTrainer, config={"env": "CartPole-v0"})

    # <class 'ray.rllib.policy.tf_policy_template.MyTFPolicy'>
    MyTFPolicy = build_tf_policy(
        name="MyTFPolicy",
        loss_fn=policy_gradient_loss)
    print()
if __name__ == "__main__":
    main()
