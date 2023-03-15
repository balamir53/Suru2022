from random import randint,random
import copy
import numpy as np
from utilities import *
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import run_experiments, register_env
from agents.GolKenari import GolKenari
from agents.TruckMini import TruckMini
from agents.RiskyValley import RiskyValley
from IndependentLearnerAll import IndependentLearnerAll
from agents.MyLearner import MyLearner
from argparse import Namespace
from models.action_mask_model import TorchActionMaskModel
import pickle
import yaml

map="TrainSingleMixedSmall"

def read_hypers():
    with open(f"/workspaces/Suru2022/data/config/{map}.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict
# def my_env_creator(args, agents):
#     return IndependentLearnerAll(args, agents)
class SelfPlayAll:
    def __init__(self, team, action_lenght):
        args = Namespace(map=map, render=False, gif=False, img=False)

        self.configs = read_hypers()
        self.agents = []
        self.truckID =0
        self.tanklID=0
        self.tankhID=0
        self.droneID=0
        for x in self.configs['blue']['units']:
            if x['type'] == 'Truck':
                self.agents.append('truck'+str(self.truckID))
                self.truckID +=1
            elif x['type'] == 'LightTank':
                self.agents.append('tankl'+str(self.tanklID))
                self.tanklID +=1
            elif x['type'] == 'HeavyTank':
                self.agents.append('tankh'+str(self.tankhID))
                self.tankhID +=1
            elif x['type'] == 'Drone':
                self.agents.append('drone'+str(self.droneID))
                self.droneID +=1

        ray.init()

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
             "multiagent": {
                "policies": {"truck", "tankl","tankh", "drone"},
                # "policies": {"truck"},
                "policy_mapping_fn": policy_mapping_fn,                    
            }
            }
            
        # self.env = my_env_creator(args, self.agents)
        register_env("ray", lambda config : IndependentLearnerAll(args, self.agents))

        ppo_agent = PPOTrainer(config=config, env="ray")
        ppo_agent.restore(checkpoint_path="/workspaces/Suru2022/models/checkpoint_000150/checkpoint-150")
       
        self.truck_pol = ppo_agent.get_policy('truck')
        self.tankl_pol = ppo_agent.get_policy('tankl')
        self.tankh_pol = ppo_agent.get_policy('tankh')
        self.drone_pol = ppo_agent.get_policy('drone')

        self.env = ppo_agent.workers.local_worker().env
        self.env.reset()
        self.firstTime = True
    def action(self, raw_state):
        # process observations
        obs_d, info = self.env._decode_state(raw_state,1)
        if not self.firstTime:
            # update self.env.agents
            self.env._decode_state(raw_state,2)
        self.firstTime = False
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
        action_to_play = self.env.apply_action(action, raw_state, self.env.team)

        if raw_state['turn'] == raw_state['max_turn']:
            self.env.reset()
            self.firstTime = True
        
        return action_to_play