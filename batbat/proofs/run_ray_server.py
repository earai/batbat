from time import sleep

import ray
import ray.rllib.agents.ppo as ppo

ray.shutdown()
ray.init(ignore_reinit_error=True)

print("Dashboard URL: http://{}".format(ray.get_webui_url()))

while True:
    pass