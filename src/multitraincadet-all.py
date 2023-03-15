import argparse
import ray
from ray import tune
from ray.rllib.policy.policy import PolicySpec
import ray.rllib.agents.ppo as ppo
from gym import spaces
import numpy as np
import os
from ray.tune import run_experiments, register_env
# from agents.GolKenari import GolKenari
from agents.IndependentLearnerAll import IndependentLearnerAll

from models.action_mask_model import TorchActionMaskModel

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='Cadet Agents')
parser.add_argument('map', metavar='map', type=str,
                    help='Select Map to Train')
parser.add_argument('--mode', metavar='mode', type=str, default="Train",
                    help='Select Mode[Train,Sim]')
parser.add_argument('--agentBlue', metavar='agentBlue', type=str,
                    help='Class name of Blue Agent')
parser.add_argument('agentRed', metavar='agentRed', type=str,
                    help='Class name of Red Agent')
parser.add_argument('--numOfMatch', metavar='numOfMatch', type=int, nargs='?', default=1,
                    help='Number of matches to play between agents')
parser.add_argument('--render', action='store_true',
                    help='Render the game')
parser.add_argument('--gif', action='store_true',
                    help='Create a gif of the game, also sets render')
parser.add_argument('--img', action='store_true',
                    help='Save images of each turn, also sets render')

args = parser.parse_args()

def main():
    ray.init()
    
    # agents are derived from map in learner class
    # hand over an empty list for now
    agents = []

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id[:5] == "truck":
            return "truck"
        elif agent_id[:5] == "tankh":
             return "tankh"
        elif agent_id[:5] == "tankl":
             return "tankl"
        elif agent_id[:5] == "drone":
             return "drone"

    register_env("ray", lambda helehele: IndependentLearnerAll(args,agents))

    # register_env("ray", lambda config: IndependentLearner(args, agents))
    config= {
            "use_critic": True,
            "log_level": "WARN",
             "num_workers": 12,
            #  "num_gpus":1,
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
                # "policies":set(env.env.agents), # first env is the group agent, seconde one independent agent
                "policies": {"truck", "tankl","tankh", "drone"},
                "policy_mapping_fn": policy_mapping_fn,                    
            }
            }
    
    run_experiments({
        "risky_ppo_recruit": {
            "run": "PPO",
            "env": "ray",
            "stop": {
                "training_iteration": 7e7,
            },
            "config": config,
            "checkpoint_freq": 50,
                # "restore": "data/inputs/model/riskyvalley/checkpoint_002800/checkpoint-2800",
        },
     },resume=True)
    # })

if __name__ == "__main__":
        main()
