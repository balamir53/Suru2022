import gym
from agents.BaseLearningGym import BaseLearningAgentGym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv
from game import Game
from gym import spaces
import yaml
import numpy as np

def read_hypers():
    with open(f"/workspaces/Suru2022/data/config/RiskyValley.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict

class IndependentLearner(MultiAgentEnv):
    def __init__(self, args, agents, team=0):
        

        # our method resembles the multiagent example in petting zoo
        # agents will be created at the start
        # but we have to figure out a way killing them and spawning new ones
        self.agents = agents
        self.tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

        # agent ids are created and handed over via training script?
        # self.agentID = 0
        
        # self.dones = set()

        # creating our game which will run a single environment that will be 
        # played via our agents 
        agentos = [None, "RandomAgent"]
        self.game = Game(args, agentos)

        # parameters for our game
        self.train = 0
        self.team = team
        self.enemy_team = 1        
        self.configs = read_hypers()
        self.height = self.configs['map']['y']
        self.width = self.configs['map']['x']
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None


        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(685,),
            dtype=np.int16
        )
        self.action_space = spaces.Discrete(7)
        # this has to be defined
        # make it smaller by processing the observation space
        # this is the next step
        # obs spaces per agent has to be created with identical matrices
        # since truck agents for now have same space
        # will these be defined at the group agent, here or in both
        # check this
        self.observation_spaces = {}
        for x in self.agents:
            self.observation_spaces[x] = self.observation_space
        # self.observation_space = spaces.Box(
        #     low=-2,
        #     high=401,
        #     shape=(2,24*18*10+4),
        #     dtype=np.int16
        # )
        # self.observation_space = spaces.Dict (
        #     {
        #     "observations": spaces.Box(
        #     low=-2,
        #     high=401,
        #     shape=(2,24*18*10+4),
        #     dtype=np.int16
        # ),
        #     # "action_mask" : spaces.Box(0.0, 1.0, shape=self.action_space.shape) }
        #     # "action_mask" : spaces.Box(0, 1, shape=(103,),dtype=np.int8) }
        #     "action_mask" : spaces.Box(0, 1, shape=(49,),dtype=np.int8) }
        # )
        # this is defined action space for just one agent
        self.action_spaces = {}
        for x in self.agents:
            self.action_spaces[x] = self.action_space
        # self.action_space = spaces.Discrete(7)

        # no idea what this is for, keep it for now
        self.resetted = False

        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

    # is this even called?
    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        # print("setup")
    
    # not used for now
    def spawn(self):
        # spawn a new agent into the curent episode
        agentID = self.agentID
        # what it this ameka
        # we whould assign en environment to the created
        # agent ?
        # but we want to manage all grouped agents in the 
        # same environment
        # lets continue for a while
        # self.agents[agentID] = MockEnv(25)

        self.agentID += 1
        return agentID
    
    def reset(self):
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0

        # consider this in the future
        # self.manipulateMape(self.game.config,self.episodes)

        state = self.game.reset()
        self.nec_obs =state

        # nope
        # self.agents = {}

        # how and when should we use this
        # elaborate
        # self.spawn()
        self.resetted = True
        # self.dones = set()

        # we should usually keep a dictionary 
        # for every agent
        # but because we will use only one environment
        # there actually one observation
        # which will be processed into separate obs for 
        # each agent
        # obs = {}
        # for i,a in self.agents.items():
        #     obs[i] = a.reset()
        
        # because we keep several agents here
        # multiagentenv expects several obs

        # obs = {}
        # for i,a in self.agents:
        #     obs[i] = a.observation_space.sample()
        obs_samples = {}
        for x in self.agents:
            obs_samples[x] = self.observation_spaces[x].sample()
        # return self.observation_space.sample()
        return obs_samples
    def _decode_state(self, obs):
        turn = obs['turn']
        max_turn = obs['max_turn'] 
        units = obs['units']
        hps = obs['hps']
        bases = obs['bases']
        score = obs['score']
        res = obs['resources']
        load = obs['loads']
        terrain = obs["terrain"]
        y_max, x_max = res.shape
        my_units = []
        enemy_units = []
        resources = []
        for i in range(y_max):
            for j in range(x_max):
                if units[self.team][i][j]<6 and units[self.team][i][j] != 0:
                    my_units.append(
                    {   
                        'unit': units[self.team][i][j],
                        'tag': self.tagToString[units[self.team][i][j]],
                        'hp': hps[self.team][i][j],
                        'location': (i,j),
                        'load': load[self.team][i][j]
                    }
                    )
                if units[self.enemy_team][i][j]<6 and units[self.enemy_team][i][j] != 0:
                    enemy_units.append(
                    {   
                        'unit': units[self.enemy_team][i][j],
                        'tag': self.tagToString[units[self.enemy_team][i][j]],
                        'hp': hps[self.enemy_team][i][j],
                        'location': (i,j),
                        'load': load[self.enemy_team][i][j]
                    }
                    )
                if res[i][j]==1:
                    resources.append((i,j))
                if bases[self.team][i][j]:
                    my_base = (i,j)
                if bases[self.enemy_team][i][j]:
                    enemy_base = (i,j)
        
        # print(my_units)
        unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        ress = [*res.reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        terr = [*terrain.reshape(-1).tolist()]
        
        state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr)
        '''
        state actually turns here into observation space for the model 
        we will decrease it for the truck agent
        current model:
        scores(2) [model doesnt get any reward for these] -not needed
        turn (1) [no model effect] -not needed (we can apply neg rew for delays)
        max_turn (1) [no model effect]
        unitss (map size * 2) - needed, but it can be reshaped as the agent coordinates and relative distance to others?
        hpss (map size * 2)  - not needed for the initial model
        basess (map size * 2) - size too much, can be modeled as coordinates?
        ress (map size) - needed, but it can be reshaped as coordinates as well?
        loads (map size * 2) - we don't need all the loads, can keep only agent truck load?
        terr (map size) - needed
        TOTAL: 4324
        '''
        '''
        ********** new model shape *********
        agents coordinate (2)
        its load (1)
        other units and  our base relative distance vectors ( 2 * (#our base + all units-1) + type) (3) 
            !! use padding for max units size like 50 (fixed to 50 * 3 = 150)
        relative distances to closest (say 50) resources (2 * resources) (100)
        terrain (map size) ? how to manage this ? cant go into water, but we dont apply any neg rew? should we?
            !! we can keep 7x7 grid for the agent staying in the center
        TOTAL: 302
        [has to be calculated per agent, all agents observations will be then handed to the model as separate obs]
        [we need also calculate reward per agent]
        '''
        return np.array(state, dtype=np.int16), (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)
    def step(self, action_dict):
        # wait a little bit
        # self.action_mask = np.ones(49,dtype=np.int8)

        harvest_reward, kill_reward, martyr_reward = 0

        # self.env.step(action[self.env.agent_selection])

        # we are expectin an action dictionary of agents
        action = np.array([x for x in action_dict.values()])

        # here we will convert the action space
        # into the game inputs as location, movement, target and train
        # game step returns next_state,reward,done

        movement = action.tolist()

        enemy_order = []

        obs_d = {}
        rew_d = {}
        done_d = {}
        info_d = {}
        while self.env.agents:
            obs, rew, done, info = self.env.last()
            a = self.env.agent_selection
            obs_d[a] = obs
            rew_d[a] = rew
            done_d[a] = done
            info_d[a] = info
            if self.env.dones[self.env.agent_selection]:
                self.env.step(None)
            else:
                break

        all_done = not self.env.agents
        done_d["__all__"] = all_done

        return obs_d, rew_d, done_d, info_d


