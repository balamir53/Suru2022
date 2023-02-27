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
from agents.MyLearner import MyLearner
from argparse import Namespace
from models.action_mask_model import TorchActionMaskModel
import pickle

class PatchedPPOTrainer(ray.rllib.agents.ppo.PPOTrainer):

    #@override(Trainable)
    def load_checkpoint(self, checkpoint_path: str) -> None:
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        worker = pickle.loads(extra_data["worker"])
        worker = PatchedPPOTrainer.__fix_recursively(worker)
        extra_data["worker"] = pickle.dumps(worker)
        self.__setstate__(extra_data)

    def __fix_recursively(data):
        if isinstance(data, dict):
            return {key: PatchedPPOTrainer.__fix_recursively(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [PatchedPPOTrainer.__fix_recursively(value) for value in data]
        elif data is None:
            return 0
        else:
            return data

class SelfPlay:
    def __init__(self, team, action_lenght):
        # args = Namespace(map="RiskyValley", render=False, gif=False, img=False)
        args = Namespace(map="RiskyValley", render=False, gif=False, img=False)
        agents = [None, "RandomAgent"]

        self.team = 0
        self.enemy_team = 1
        self.action_mask = np.ones(49,dtype=np.int8)
    
        # ray.init(num_gpus=1, log_to_driver=True)
        ray.init()
        # config= {"use_critic": True,
        #      "num_workers": 1,
        #      "use_gae": True,
        #      "lambda": 1.0,
        #      "kl_coeff": 0.2,
        #      "rollout_fragment_length": 200,
        #      "train_batch_size": 4000,
        #      "sgd_minibatch_size": 128,
        #      "shuffle_sequences": True,
        #      "num_sgd_iter": 30,
        #      "lr": 5e-5,
        #      "lr_schedule": None,
        #      "vf_loss_coeff": 1.0,
        #      "framework": "torch",
        #      "entropy_coeff": 0.0,
        #      "entropy_coeff_schedule": None,
        #      "clip_param": 0.3,
        #      "vf_clip_param": 10.0,
        #      "grad_clip": None,
        #      "kl_target": 0.01,
        #      "batch_mode": "truncate_episodes",
        #      "observation_filter": "NoFilter"}
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
                }
            }
        # register_env("ray", lambda config: RiskyValley(args, agents))
        register_env("ray", lambda config: MyLearner(args, agents))
        ppo_agent = PPOTrainer(config=config, env="ray")
        # ppo_agent = PatchedPPOTrainer(config=config, env="ray")
        # ppo_agent = PPOTrainer(env="ray")
        # ppo_agent.restore(checkpoint_path="data/inputs/model/checkpoint_002600/checkpoint-2600") # Modelin Bulunduğu yeri girmeyi unutmayın!
        # ppo_agent.restore(checkpoint_path="data/inputs/model/truckmini/checkpoint_000850/checkpoint-850")
        # ppo_agent.restore(checkpoint_path="data/inputs/model/riskyvalley/minimixed/checkpoint_002400/checkpoint-2400")
        ppo_agent.restore(checkpoint_path="/workspaces/Suru2022/models/checkpoint_001350/checkpoint-1350")
        # ppo_agent.restore(checkpoint_path="models/checkpoint_000005/checkpoint-5") # Modelin Bulunduğu yeri girmeyi unutmayın!
        self.policy = ppo_agent.get_policy()

    def action(self, raw_state):
        '''
        pos=[3, 17]
        target=[10, 15]
        astar(pos,target,state)
        return
        '''
        self.action_mask = np.ones(49,dtype=np.int8)
        #TODO: get the state from already loaded checkpoint
        # state = RiskyValley.just_decode_state(raw_state, self.team, self.enemy_team)
        state, info = MyLearner.just_decode_state_(raw_state, self.team, self.enemy_team)
        self.x_max, self.y_max, self.my_units, self.enemy_units, self.resources, self.my_base, self.enemy_base = info

        for i,unit in enumerate(self.my_units):
            if (i>6):
                break
            if(unit['tag']!='Truck'):
                continue
            # check first if its loaded and on the base
            if (unit['location']==base) and unit['load']>0:
                # find the index of the truck in the unit list
                self.action_mask[i*7]=1
                # mask actions other than 0
                self.action_mask[i*7+1:i*7+7]=0
                continue
            if (unit['load']>2):
                continue
            for reso in self.resources:            
                # if there is resource on the next location of the truck
                if (reso == unit['location']):
                        # find the index of the truck in the unit list
                        self.action_mask[i*7]=1
                        # mask actions other than 0
                        self.action_mask[i*7+1:i*7+7]=0

        actions, _, _ = self.policy.compute_single_action({"observations":state.astype(np.float32),"action_mask":self.action_mask})
        
        #TODO: get the state from already loaded checkpoint
        # actions, _, _ = self.policy.compute_single_action(state.astype(np.float32))
        movement = []
        target = []
        locations = []
        counter = {"Truck":0,"LightTank":0,"HeavyTank":0,"Drone":0}
        movement = actions[0:7]
        # movement = multi_forced_anchor(movement, raw_state, self.team)
        movement = movement.tolist()
        while len(movement) > len(self.my_units):
            movement.pop()
        
        # TODO: Write location
        for unit in self.my_units:
            locations.append(unit['location'])
        
        # TODO: Write target
        enemy_locs_ = []
        for e_unit in self.enemy_units:
            enemy_locs_.append(e_unit['location'])
            
        nearest_enemy_locs = []
        for unit in self.my_units:
            counter[unit['tag']]+=1
            if len(self.enemy_units) == 0 or len(self.enemy_units) < 0:
                break
            nearest_enemy_locs.append(nearest_enemy(unit['location'], enemy_locs_))
        
        if 0 == len(self.my_units):
            locations = []
            movement = []
            target = []
            train = randint(1,4)
            return (locations, movement, target, train)
        elif 0 < len(self.my_units) <= 7:
            ally_count = len(self.my_units)
        
            if len(self.enemy_units) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)
            
            while len(enemy_order) > ally_count:
                enemy_order.pop()
        
        elif len(self.my_units) > 7:
            ally_count = 7

            if len(self.enemy_units) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)
            
            ##added by luchy:due to creating nearest enemy locs for each ally, if number of allies are over 7, only 7 targets must be defined.
            enemy_order = enemy_order[:7]
            
            while len(locations) > 7:
                locations = list(locations)[:7]
        
        # if the distance between ally and enemy is less than 3 then movement will be 0 as a preparation to shoot.
        # for i in range(len(locations)):
        #     if getDistance(locations[i], enemy_order[i]) <= [self.x_max, self.y_max][np.argmin([self.x_max, self.y_max])] and self.my_units[i]["tag"] != "Truck":
        #     # if getDistance(locations[i], enemy_order[i]) <= 3 and self.my_units[i]["tag"] != "Truck":
        #         movement[i] = 0
        
        locations = list(map(tuple, locations))
        target = list(map(tuple, enemy_order))
        
        # TODO: Write train logic
        train = 0

        number_of_tanks, number_of_enemy_tanks, number_of_uavs, number_of_enemy_uavs, number_of_trucks, number_of_enemy_trucks = 0, 0, 0, 0, 0, 0

        for x in self.my_units:
            if x["tag"] == "HeavyTank" or x["tag"] == "LightTank":
                number_of_tanks+=1
            elif x["tag"] == "Drone":
                number_of_uavs+=1
            elif x["tag"] == "Truck":
                number_of_trucks+=1
        for x in self.enemy_units:
            if x["tag"] == "HeavyTank" or x["tag"] == "LightTank":
                number_of_enemy_tanks+=1
            elif x["tag"] == "Drone":
                number_of_enemy_uavs+=1
            elif x["tag"] == "Truck":
                number_of_enemy_trucks+=1
        
        number_of_our_military = number_of_tanks+number_of_enemy_uavs
        number_of_enemy_military =number_of_enemy_tanks+number_of_enemy_uavs
        
        if raw_state["score"][self.team]>raw_state["score"][self.enemy_team]+2:
            if counter["Truck"]<2:
                train = stringToTag["Truck"]
            elif counter["LightTank"]<1:
                train = stringToTag["LightTank"]
            elif counter["HeavyTank"]<1:
                train = stringToTag["HeavyTank"]
            elif counter["Drone"]<1:
                train = stringToTag["Drone"]
            # elif len(self.my_units)<len(self.enemy_units):
            elif number_of_our_military<number_of_enemy_military:
                train = randint(2,4)
        elif counter["Truck"] < 1:
            train = stringToTag["Truck"]
        elif raw_state["score"][self.team]+2<raw_state["score"][self.enemy_team] and len(self.my_units)<len(self.enemy_units)*2:
            train = randint(2,4)
        
        return (locations, movement, target, train)
        # train = 1
        # location, movement, target, train = MyLearner.just_take_action(actions, raw_state, self.team, train)        
        # return (location, movement, target, train)