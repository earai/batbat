from ray.rllib import MultiAgentEnv

MultiAgentTrafficEnv(MultiAgentEnv)

env = MultiAgentTrafficEnv(num_cars=20, num_traffic_lights=5)