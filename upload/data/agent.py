# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#     print("done")

# install('gym==0.21.0')
# install('astar')
from pip._internal.operations import freeze
pkgs = freeze.freeze()
for pkg in pkgs: print(pkg)

from astar import AStar
from action_mask_model import TorchActionMaskModel
from IndependentLearnerAll import IndependentLearnerAll
from ray.tune import register_env
from argparse import Namespace
from ray.rllib.agents.ppo import PPOTrainer
import numpy as np
class EvaluationAgent():

    def __init__(self, observation_space, action_space):
        self.action_space = action_space

        self.agents = []

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id[:5] == "truck":
                return "truck"
            elif agent_id[:5] == "tankh":
                return "tankh"
            elif agent_id[:5] == "tankl":
                return "tankl"
            elif agent_id[:5] == "drone":
                return "drone"
            
        config= {"use_critic": True,
            "log_level": "WARN",
             "num_workers": 0,
             "use_gae": True,
             "lambda": 1.0,
             "kl_coeff": 0.2,
             "rollout_fragment_length": 200,
             "train_batch_size": 1280,
             "sgd_minibatch_size": 128,
             "shuffle_sequences": True,
             "num_sgd_iter": 30,
             "lr": 5e-5,
             "lr_schedule": None,
             "vf_loss_coeff": 1.0,
             "framework": "torch",
             "entropy_coeff": 0.0,
             "entropy_coeff_schedule": None,
             "clip_param": 0.3,
             "vf_clip_param": 10.0,
             "grad_clip": None,
             "kl_target": 0.01,
             "batch_mode": "truncate_episodes",
             "observation_filter": "NoFilter",
             "model":{
                "custom_model": TorchActionMaskModel
             },
             "multiagent": {
                "policies": {"truck", "tankl","tankh", "drone"},
                # "policies": {"truck"},
                "policy_mapping_fn": policy_mapping_fn,                    
            }
            }
        args = Namespace(map=map, render=False, gif=False, img=False, mapChange=False)
        register_env("ray", lambda config : IndependentLearnerAll(args, self.agents,mapChange=args.mapChange))
        ppo_agent = PPOTrainer(config=config, env="ray")
        # TODO :change this to relative path
        ppo_agent.restore(checkpoint_path="model/checkpoint_002250/checkpoint-2250")
       
        self.truck_pol = ppo_agent.get_policy('truck')
        self.tankl_pol = ppo_agent.get_policy('tankl')
        self.tankh_pol = ppo_agent.get_policy('tankh')
        self.drone_pol = ppo_agent.get_policy('drone')

        self.env = ppo_agent.workers.local_worker().env
        # self.env.reset()
        self.firstTime = True
        self.init_once = True

    def act(self, observation):
        
        if self.init_once:
            self.env.late_init(observation)

        if not self.firstTime:
            # update self.env.agents
            self.env._decode_state(observation,2)

        # process observations
        obs_d, info = self.env._decode_state(observation,1)

        self.firstTime = False
        self.init_once = False
        # get actions
        self.env.current_action = {}
        
        for x in self.env.agents:
            policy = None
            if x[:5] == "truck":
                policy = self.truck_pol
            elif x[:5] == "tankh":
                policy = self.tankh_pol
            elif x[:5] == "tankl":
                policy = self.tankl_pol
            elif x[:5] == "drone":
                policy = self.drone_pol
            action, _, _ = policy.compute_single_action(obs_d[x])
            self.env.current_action[x] = action
        
        action = np.array([x for x in self.env.current_action.values()])
        action_to_play = self.env.apply_action(action, observation, self.env.team)

        if observation['turn'] == observation['max_turn']:
            self.env.reset()
            self.firstTime = True
        
        return action_to_play
    

