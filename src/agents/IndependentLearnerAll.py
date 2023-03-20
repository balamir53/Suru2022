import gym
import math
import copy
from agents.BaseLearningGym import BaseLearningAgentGym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv
from game import Game
from gym import spaces
import yaml
import numpy as np
import random
from utilities import ally_locs, enemy_locs, nearest_enemy_selective, getMovement, getDirection, getDistance, tagConverter

def read_hypers(map):
    with open(f"/workspaces/Suru2022/data/config/{map}.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict
UNITS_PADDING = 50*3 # parameter * (y,x and type)
RESOURCE_PADDING = 50*2 # parameter * (y and x)
TERRAIN_PADDING = 7*7 # parameter
# update this in init function for smaller maps
MAX_DISTANCE = 30
class IndependentLearnerAll(MultiAgentEnv):
    def __init__(self, args, agents, team=0, mapChange=False):
        # agents is an empty list to be filled
        self.agents = agents
        self.mapChange = mapChange
        # get agents from map config
        self.configs = read_hypers(args.map)
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

        self.init_truckID = copy.copy(self.truckID)
        self.init_tanklID = copy.copy(self.tanklID)
        self.init_tankhID = copy.copy(self.tankhID)
        self.init_droneID = copy.copy(self.droneID)

        # our method resembles the multiagent example in petting zoo
        # agents will be created at the start
        # but we have to figure out a way killing them and spawning new ones
        
        self.init_agents = copy.copy(agents)
        self.tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

        # keep it simple for now
        # self.unit_type = {
        #     'base': 0,
        #     'my_truck': 1,
        #     'my_ltank': 2,
        #     'my_htank': 3,
        #     'my_drone': 4,
        #     'en_truck': 5,
        #     'en_ltank': 6,
        #     'en_htank': 7,
        #     'en_drone': 8
        # }
        self.unit_type = {
            'base': 1,
            'friend': 2,
            'foe': 3
        }
        # agent ids are created and handed over via training script?
        # self.agentID = 0
        
        # self.dones = set()

        # creating our game which will run a single environment that will be 
        # played via our agents 

        self.game = Game(args, [None, args.agentRed])

        # parameters for our game
        self.train = 0
        self.team = team
        self.enemy_team = 1        
        
        self.height = self.configs['map']['y']
        self.width = self.configs['map']['x']
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None

        self.mapChangeFrequency = 1

        if self.height < 18:
            MAX_DISTANCE = int(math.sqrt(self.height**2+self.width**2))

        self.load_reward = 0.5
        self.unload_reward = 1
        self.kill_reward = 1
        self.pos_partial = 0.05
        self.neg_partial = -0.01
        self.stuck_reward = -10
        self.stuck_agents = []
        self.current_action = {}
        self.old_my_units = {}
        self.dead_units = []
        self.dead_ones = set()
        # self.observation_space = spaces.Box(
        #     low=-40,
        #     high=401,
        #     shape=(302,),
        #     dtype=np.int16
        # )

        self.observation_space = spaces.Dict(
            {
            "observations": spaces.Box(
            low=-40,
            high=401,
            shape=(302,),
            dtype=np.int16
        ),
            "action_mask": spaces.Box(0, 1, shape=(7,), dtype=np.int8)
            }
        )
        # 0 is stay/fire/load/unload
        # other 6 are directions
        self.action_space = spaces.Discrete(7)
        # this has to be defined
        # make it smaller by processing the observation space
        # this is the next step
        # obs spaces per agent has to be created with identical matrices
        # since truck agents for now have same space
        # will these be defined at the group agent, here or in both
        # check this
        self.observation_spaces = {}
        self.obs_dict = {}
        self.loads = {}
        self.rewards = {}
        self.dones = {}
        self.infos = {}
        self.action_masks = {}
        # to give partial rewards to trucks on their distances to base
        self.old_base_distance = {}
        self.old_my_units = {}
        for x in self.agents:
            self.observation_spaces[x] = self.observation_space
            self.obs_dict[x] = {"observations":[], "action_mask":[]}
            self.loads[x] = 0
            self.rewards[x] = 0
            self.dones[x] = False  #if agents die make this True
            self.infos[x] = {}
            self.old_base_distance[x] = 30
            self.action_masks[x] = np.ones(7, dtype=np.int8)
        self.dones['__all__'] = False
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

        self.agents_positions = {}
        # get the initial positions of agents
        # should we apply the same logic as in decode state?
        for i in range(len(self.agents)):
            self.agents_positions[self.agents[i]]=(self.configs['blue']['units'][i]['y'], self.configs['blue']['units'][i]['x'])
        self.my_base = (self.configs['blue']['base']['y'],self.configs['blue']['base']['x'])
        
        # gets terrain locs [(location(x,y) ,terrain_type)] --> terrain_type : 'dirt' : 1, 'water' : 2, 'mountain' : 3}
        self.terrain = self.terrain_locs()
        self.init_resource_num = len(self.configs["resources"])
        self.resources = []
        self.old_raw_state = None
        self.firstShot = True

    def getCoordinate(self, dict):
        return dict['y']*self.width+dict['x']

    def addOffSet(self, dict, xOff, yOff):
        dict['x']+= xOff
        dict['y']+= yOff

    def resetPosition(self, myDict):
        myDict['blue']['base']['x'] = self.configs['blue']['base']['x']
        myDict['blue']['base']['y'] = self.configs['blue']['base']['y']

        myDict['red']['base']['x'] = self.configs['red']['base']['x']
        myDict['red']['base']['y'] = self.configs['red']['base']['y']

        for i in range(len(myDict['blue']['units'])):
            myDict['blue']['units'][i]['x'] = self.configs['blue']['units'][i]['x']
            myDict['blue']['units'][i]['y'] = self.configs['blue']['units'][i]['y']

        for i in range(len(myDict['red']['units'])):
            myDict['red']['units'][i]['x'] = self.configs['red']['units'][i]['x']
            myDict['red']['units'][i]['y'] = self.configs['red']['units'][i]['y']

    def manipulateMap(self, mapDict, episode):
        # here we manipulate actually self.game.config
        # change resources positions on every episode
        
        # mapDict['blue']['base']['x'] = 0 #this works
        
        # mapDict = copy.deepcopy(self.configs)
        # mapDict = self.configs.copy() #this doesnt work
        # mapDict['blue']['base']['x'] = 0
        xOffSet = 0
        yOffSet = 0
        # change the base and units' first positions on some frequency
        # if(episode%self.mapChangeFrequency==0):
        # if(False):
            # print(episode)
            # self.resetPosition(mapDict)
            # xOffSet = random.randint(0,self.width-self.gameAreaX)
            # yOffSet = random.randint(0,self.height-self.gameAreaY)
            # self.addOffSet(mapDict['blue']['base'],xOffSet, yOffSet)
            # self.addOffSet(mapDict['red']['base'],xOffSet, yOffSet)
            # for x in mapDict['blue']['units']:
            #     self.addOffSet(x,xOffSet, yOffSet)
            # for x in mapDict['red']['units']:
            #     self.addOffSet(x,xOffSet, yOffSet)

        if(episode%self.mapChangeFrequency==0):
            # random base on the most left tile column
            new_base_y = random.randint(0,self.height-1)
            mapDict['blue']['base']['y'] = new_base_y
            self.my_base = (new_base_y, 0)
            # find out already occupied tiles
            occupiedTiles = {self.getCoordinate(mapDict['blue']['base']), self.getCoordinate(mapDict['red']['base'])}
            for x in mapDict['blue']['units']:
                occupiedTiles.add(self.getCoordinate(x))
            for x in mapDict['red']['units']:
                occupiedTiles.add(self.getCoordinate(x))

            if self.terrain:
                for ter in self.terrain.keys():
                   occupiedTiles.add(ter)
                
            # randomize resource positions
            for x in mapDict['resources']:
                a = random.randint(0, self.width-1)+xOffSet
                b = random.randint(0, self.height-1)+yOffSet
                while self.getCoordinate({'x':a,'y':b}) in occupiedTiles:
                    a = random.randint(0, self.width-1)+xOffSet
                    b = random.randint(0, self.height-1)+yOffSet
                occupiedTiles.add(self.getCoordinate({'x':a,'y':b}))
                x['x'] = a
                x['y'] = b

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
        self.agents = copy.copy(self.init_agents)
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0

        # consider this in the future
        if self.mapChange:
            self.manipulateMap(self.game.config,self.episodes)

        state = self.game.reset()
        self.nec_obs =state

        # nope
        # self.agents = {}
        self.observation_spaces = {}
        self.obs_dict = {}
        self.loads = {}
        self.rewards = {}
        self.dones = {}
        self.infos = {}
        self.old_base_distance = {}
        self.action_masks = {}
        for x in self.agents:
            self.observation_spaces[x] = self.observation_space
            self.obs_dict[x] = {"observations":[], "action_mask":[]}
            self.loads[x] = 0
            self.rewards[x] = 0
            self.dones[x] = False  #if agents die make this True
            self.infos[x] = {}
            self.old_base_distance[x] = 30
            self.dones[x] = False
            self.action_masks[x] = np.ones(7, dtype=np.int8)

        self.agents_positions = {}
        for i in range(len(self.agents)):
            self.agents_positions[self.agents[i]]=(self.configs['blue']['units'][i]['y'], self.configs['blue']['units'][i]['x'])
        
        self.truckID=copy.copy(self.init_truckID)
        self.tanklID=copy.copy(self.init_tanklID)
        self.tankhID=copy.copy(self.init_tankhID)
        self.droneID=copy.copy(self.init_droneID)
        # clear the dead ones set
        self.dead_ones.clear()
        self.dead_units = []
        # how and when should we use this
        # elaborate
        # self.spawn()
        self.resetted = True
        # self.dones = set()
        self.dones['__all__'] = False
        self.stuck_agents = []
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
    
    def load_reward_check(self, old_load, new_load, truck_id):
        # apply reward for collecting loads
        if new_load > old_load and old_load<3:
            self.rewards[truck_id] += self.load_reward
        # if it unloads the the loads on base
        if self.agents_positions[truck_id] == self.my_base and old_load>new_load:
            self.rewards[truck_id] += self.unload_reward * old_load
        pass
    
    def kill_reward_check(self, obs):
        old_enemy, new_enemy = self.enemy_unit_dicts(obs, self.team)
        for _ , old in enumerate(old_enemy):
            for _, new in enumerate(new_enemy):
                if (old["location"] == new["location"]) and old["tag"] != "Dead" and new["tag"] == "Dead":
                    for k, shoot_loc in enumerate(self.nearest_enemy_locs):
                        #TODO: self.current_action[self.agents[k] ] is not safe check this.
                        if tuple(shoot_loc) == old["location"] and self.current_action[self.agents[k]] == 0:
                            self.rewards[self.agents[k]] += self.kill_reward
    
    def tank_stuck_reward_check(self, x):
        # gets terrain locs [(location(x,y) ,terrain_type)] --> terrain_type : 'dirt' : 1, 'water' : 2, 'mountain' : 3}
        # for i,x in enumerate(self.agents_positions):
        #     if x[:5] != "tankh" or x in self.stuck_agents:
        #         continue
        agent_pos_key = self.agents_positions[x][0]*self.width+self.agents_positions[x][1]
        if self.terrain.get(agent_pos_key) == 1:
            self.rewards[x] += self.stuck_reward
            self.stuck_agents.append(x)
            #momentarily mask the action to stay.
            self.action_masks[x][1:] = 0
    
    def _spawn_agent(self):
        # if there is a unit there already, return
        for h in self.agents:
            if self.agents_positions[h] == self.my_base:
                return
        x = None
        if self.train == 1:
            x = 'truck'+str(self.truckID)
            self.agents.append(x)                    
            self.truckID +=1
        elif self.train == 2:
            x = 'tankl'+str(self.tanklID)
            self.agents.append(x)
            self.tanklID +=1
        elif self.train == 3:
            x = 'tankh'+str(self.tankhID)
            self.agents.append(x)
            self.tankhID +=1
        elif self.train == 4:
            x = 'drone'+str(self.droneID)
            self.agents.append(x)
            self.droneID +=1
        self.agents_positions[x] = self.my_base
        # it is just created and has actually no action to play
        self.current_action[x] = 0
        self.observation_spaces[x] = self.observation_space
        self.obs_dict[x] = {"observations":[], "action_mask":[]}
        self.loads[x] = 0
        self.rewards[x] = 0
        self.dones[x] = False  #if agents die make this True
        self.infos[x] = {}
        self.old_base_distance[x] = 30
        self.action_masks[x] = np.ones(7, dtype=np.int8)


    def  _decode_state(self, obs, procOrUpdate=0):
        # this function is also called from inference mode with two options
        # procOrUpdate = 1 is for obs process call from inference
        # procOrUpdate = 2 is for update agents call from inference
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
        self.resources = []
        
        if procOrUpdate != 2:
            for x in self.agents:
                self.action_masks[x] = np.ones(7,dtype=np.int8)

        #with self.nec_obs as old state, obs as new state returns old enemy and new state enemy details.
        #think whether enemy type is important or not.
        if procOrUpdate == 0:
            self.kill_reward_check(obs)
        # neg rew if the tankh is stuck on dirt.

        # wreckage time is assumed 5, we should get this from rules.yaml
        for x in self.dead_units:
            x[2] += 1
            if x[2] > 5:
                self.dead_units.remove(x)
                self.dead_ones.remove((x[0],x[1]))


        someone_just_died_at = []
        # dead_units = []
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
                elif units[self.team][i][j]==8:
                    # we assume some unit is dead
                    if self.dead_ones.isdisjoint({(i,j)}):
                        someone_just_died_at.append((i,j))
                        self.dead_ones.add((i,j))
                        # the third element is wreckage timer
                        self.dead_units.append([i,j,1])
                    # this just have been killed or it is wreckage 
                    # self.dead_units.append((i,j,0))
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
                    self.resources.append((i,j))
                if bases[self.team][i][j]:
                    my_base = (i,j)
                if bases[self.enemy_team][i][j]:
                    enemy_base = (i,j)

        # procOrUpdate = 1 is for obs process call from inference
        # procOrUpdate = 2 is for update agents call from inference
        if procOrUpdate != 1 :

            # # this is not safe check
            # # even if we set self.train greater than zero game sometimes wont train because of some limitations
            # this part has been moved to spawn_agent function
            # # elaborate
            # if self.train > 0:


            # update here self.agents and self.agents_positions
            # how to check which agent at which position has been killed?
            # print(my_units)
            # here we also check loads
            # apply reward for any load increase and decrease if on base

            someone_died_or_spawned = False
            to_be_deleted = []
            someone_has_spawned = False
            if self.train:
                if someone_just_died_at:
                    if len(self.agents_positions)-len(someone_just_died_at) != len(my_units):
                        someone_has_spawned = True
                else:
                    if len(my_units) != len(self.agents):
                        someone_has_spawned = True
                # if dead unit is on the base
                # do not spawn
                if someone_has_spawned:
                    for d in self.dead_units:
                        if (d[0],d[1]) == self.my_base:
                            someone_has_spawned = False
                            break
            for i,x in enumerate(self.agents):
                # check for deaths
                # unit is actually death on its next position
                # but maybe it didnt move because of some obstacle
                # so we should first check if it moved to the death position?
                # blue plays first
                # so it will play first and then be dead, this is important

                # what if a new unit has been created on the base
                
                # spawn the unit at the end

                
                # it will be a problem for our units to go that position
                # if someone_just_died_at:

                    # this will be true for wrackage time (5 as default) in rules.yaml
                
                # what if someone died and someone spawned at the same time
                # if len(self.agents) != len(my_units):
                #     # what if someone died and someone has spawned at the same time
                #     # maybe we can get death units from above (where they are set to 8 and wait at that position 5 steps)
                #     someone_died_or_spawned = True
                # else:
                #     someone_died_or_spawned = False
                # check if it is on the tile supposed to be
                move_x, move_y = getMovement(self.agents_positions[x],self.current_action[x])
                new_pos = tuple(map(lambda i, j: i + j, self.agents_positions[x], (move_y, move_x)))
                # check if this in my_units but this still can be wrong
                # maybe another unit moved to the locations and this one cant?
                # or there was a no-go section to go
                old_load = self.loads[x]
                # check of no-go section for lake because of the drones ---> terrain_type : 'dirt' : 1, 'water' : 2, 'mountain' : 3
                if x[:5] != 'drone' and self.terrain and self.terrain.get(new_pos[0]*self.width+new_pos[1]) == 2:
                    new_pos = self.agents_positions[x]
                    if someone_just_died_at:
                        dead = True
                        for z in my_units:
                            if z['location'] == new_pos:
                                dead = False
                                break
                        if dead:
                            to_be_deleted.append(x)
                    continue
                if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.height or new_pos[1] >= self.width:
                    new_pos = self.agents_positions[x]
                    # don't change position
                    if x[:5] == "truck":
                        self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                        self.load_reward_check(old_load, self.loads[x], x)
                    if someone_just_died_at:
                        dead = True
                        for z in my_units:
                            if z['location'] == new_pos:
                                dead = False
                                break
                        if dead:
                            to_be_deleted.append(x)
                    continue

                to_break = False
                # check if there is already a unit set there
                # we are not checking terrain and enemy units
                for y,q in enumerate(self.agents):
                    if y == i:
                        continue
                    if self.agents_positions[q] == new_pos:
                        to_break = True
                        break
                if to_break:
                    # if there is already a unit in that pos
                    # set it to old pos
                    new_pos = self.agents_positions[x]
                    if x[:5] == "truck":
                        self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                        self.load_reward_check(old_load, self.loads[x], x)
                    if someone_just_died_at:
                        dead = True
                        for z in my_units:
                            if z['location'] == new_pos:
                                dead = False
                                break
                        if dead:
                            to_be_deleted.append(x)
                    continue

                am_i_alive = False      

                # if someone_just_died_at or (len(self.agents) == len(my_units) and self.train == 0):
                # this should be entered in every case
                if True:
                    # two options, either nothing changed
                    # or a new unit has been created and another has been killed
                    # check self.train
                    for z in my_units:
                        # check for the new positions in the new state
                        # maybe i couldnt move there and there is already other agent sitting there?
                        # but we check this above?
                        if z['location'] == new_pos:
                            self.agents_positions[x] = new_pos
                            if x[:5] == "truck":
                                self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                                self.load_reward_check(old_load, self.loads[x], x) 
                            am_i_alive = True
                            break
                    # here is the case there is a either a dead unit or enemy unit on my movement direction
                    if not am_i_alive:
                        for z in my_units:
                            # set it to current pos and check if its there
                            # if it is then it is alive and we should keep it 
                            new_pos = self.agents_positions[x]
                            if z['location'] == new_pos:
                                self.agents_positions[x] = new_pos
                                if x[:5] == "truck":
                                    self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                                    self.load_reward_check(old_load, self.loads[x], x) 
                                am_i_alive = True
                                break
                if not am_i_alive and someone_just_died_at:
                    to_be_deleted.append(x)
                    # del self.agents_positions[x]
                    # del self.agents_positions_[x]
                    # self.agents.remove(x)
                    # del self.observation_spaces[x] 
                    # del self.obs_dict[x] 
                    # del self.loads[x] 
                    # del self.rewards[x] 
                    # del self.dones[x] 
                    # del self.infos[i] 
                    continue

            if to_be_deleted:
                for i in range(len(to_be_deleted)):
                    del self.agents_positions[to_be_deleted[i]]
                    self.agents.remove(to_be_deleted[i])
                    del self.observation_spaces[to_be_deleted[i]] 
                    del self.obs_dict[to_be_deleted[i]] 
                    del self.loads[to_be_deleted[i]] 
                    del self.rewards[to_be_deleted[i]] 
                    del self.dones[to_be_deleted[i]]
                    del self.old_base_distance[to_be_deleted[i]]
                    del self.infos[to_be_deleted[i]]
                    del self.action_masks[to_be_deleted[i]]
            counter = 0
            if someone_has_spawned:
                there_is_one = False
                # check if really one has been spawned
                # for agent in self.agents_positions:
                #     if self.agents_positions[agent] == self.my_base:
                #         there_is_one = True
                for h in my_units:
                    if h['location'] == self.my_base:
                        there_is_one = True
                if there_is_one:
                    # if there is a unit already on the base
                    # this function returns w/o spawning any agent
                    self._spawn_agent()
            
            wild_delete = []
            for i, agent in enumerate(self.agents_positions):
                there_is_one = False
                for uni in my_units:
                    if self.agents_positions[agent] == uni['location']:
                        there_is_one = True
                        counter +=1
                        break
                if not there_is_one:
                    wild_delete.append(agent)
            for y in wild_delete:
                del self.agents_positions[y]
                self.agents.remove(y)
                del self.observation_spaces[y] 
                del self.obs_dict[y] 
                del self.loads[y] 
                del self.rewards[y] 
                del self.dones[y]
                del self.old_base_distance[y]
                del self.infos[y]
                del self.action_masks[y]
            if counter < len(self.agents_positions):

                print(self.agents_positions)
                print(my_units)
                print('Done')

        # procOrUpdate = 2 is for update agents call from inference
        if procOrUpdate == 2 :
            return
        
        _, new_enemy = self.enemy_unit_dicts(obs, self.team)        
        # return a dict of agents obs
        for i,x in enumerate(self.agents):
            rel_dists = []
            # add rel dist to base
            my_pos = self.agents_positions[x]
            rel_dists+= (np.array(self.my_base)- np.array(my_pos)).tolist()
            rel_dists.append(self.unit_type['base'])
            
            # check for partial rewards of trucks loaded 3
            dist_to_base = np.linalg.norm(np.array(self.my_base)- np.array(my_pos))
            if x[:5] == 'truck':
                # if loaded truck is on the base force it to unload
                if my_pos == my_base and self.loads[x] > 0:
                    self.action_masks[x][1:] = 0
                # TODO: this wont work if there is an obstacle btw truck and base
                # comment it for now
                # if self.loads[x] > 2:
                #     if dist_to_base >= self.old_base_distance[x]:
                #         self.rewards[x]+= self.neg_partial
                #     else:
                #         # self.rewards[x]+= self.pos_partial
                #         self.rewards[x]+= (MAX_DISTANCE-dist_to_base)**2 / 10000
            self.old_base_distance[x] = dist_to_base

            # action mask if mud.
            if x in self.stuck_agents:
                self.action_masks[x][1:] = 0
            
            # rel dist to friendly units
            for y in my_units:
                # if itself, skip
                if my_pos == y['location']:
                    continue                
                rel_dists+=(np.array(y['location'])- np.array(my_pos)).tolist()
                rel_dists.append(self.unit_type['friend'])
            # rel dist to enemy units
            for y in enemy_units:
                rel_dists+=(np.array(y['location'])- np.array(my_pos)).tolist()
                rel_dists.append(self.unit_type['foe'])
            # add padding for the rest
            rel_dists+=[0]*(UNITS_PADDING-len(rel_dists))

            # check rel distances to resources and take first 50 into account
            res_dists = []
            sorted_dist = []
            for z,y in enumerate(self.resources):
                # if a truck is on a resource force it to collect
                if x[:5] == 'truck' and my_pos == y:
                    self.action_masks[x][1:] = 0
                # get the distances and indexes as tuple
                dis = int(np.linalg.norm(np.array(y)-np.array(my_pos)))
                sorted_dist.append((dis,z))
            # sort the distances with the indexed
            sorted_dist = sorted(sorted_dist, key= lambda x: x[0])
            # get relative distances of the first RESOURCE_PADDING/2
            index = 0
            if len(sorted_dist)>RESOURCE_PADDING/2:
                index = int(RESOURCE_PADDING/2)
            else: 
                index = len(sorted_dist)
            for z in range(index):
                res_dists+= (np.array(self.resources[sorted_dist[z][1]])- np.array(my_pos)).tolist()
            # pad the remaining res distances if its shorter than RESOURCE PADDING
            if len(res_dists)<RESOURCE_PADDING:
                res_dists+=[0]*(RESOURCE_PADDING-len(res_dists))
            
            # get the terraing around the agent
            agent_surround = [0] * TERRAIN_PADDING 
            if self.terrain:
                # agent_pos = np.array(self.agents_positions[i])
                counter = 0
                # check for surround terrain in TERRAIN_PADDING box, if there is change agent surround index accordingly with terrain type
                # check of no-go section for whole unit types ---> terrain_type : 'dirt' : 1, 'water' : 2, 'mountain' : 3
                index = int(math.sqrt(TERRAIN_PADDING)//2)
                for ver in range(-index, index+1):
                    for hor in range(-index, index+1):
                        coor = (my_pos[0] + ver, my_pos[1] + hor)
                        lookat = coor[0]*self.width + coor[1]
                        # mask drone's action to not to go to mountain-side.
                        if x[:5] == 'drone' and abs(ver)<2 and abs(hor)<2 and self.terrain.get(lookat) == 3: 
                            direction = getDirection(my_pos[1], hor, ver)
                            if direction < 7:
                               self.action_masks[x][direction] = 0 
                        # mask tank and truck' action to not to go to mountain-side, water.
                        if (x[:4] == 'tank' or x[:5] == 'truck') and abs(ver)<2 and abs(hor)<2 and (self.terrain.get(lookat) == 2 or self.terrain.get(lookat) == 3): 
                            direction = getDirection(my_pos[1], hor, ver)
                            if direction < 7:
                               self.action_masks[x][direction] = 0 
                        if self.terrain.get(lookat):
                            agent_surround[counter] = self.terrain[lookat]
                        counter += 1
            #fire action mask for tankh, tankl, and drone.
            if x[:5] != 'truck':
                tag = tagConverter(x)
                nearest_enemy = nearest_enemy_selective({"tag" : tag, "location" : self.agents_positions[x]}, new_enemy)
                if (x[:4] =='tank') and nearest_enemy and getDistance(self.agents_positions[x], nearest_enemy["location"]) < 4:
                    self.action_masks[x][1:] = 0
                elif (x[:5] =='drone') and nearest_enemy and getDistance(self.agents_positions[x], nearest_enemy["location"]) < 2:
                    self.action_masks[x][1:] = 0
            # this should also in update
            # TODO WTF ?
            if self.terrain and x[:5] == "tankh" and x not in self.stuck_agents:
                self.tank_stuck_reward_check(x)
            
            # if unit is on yth y position,it cannot go down anymore, mask actions 4,5,6
            if self.agents_positions[x][0] == (self.height-1) :
                if self.agents_positions[x][1] % 2:
                   self.action_masks[x][5] = 0 
                else:
                   self.action_masks[x][4:] = 0
            # if unit is on 0th y position, it cannot go up anymore, mask actions 1,2,3
            elif self.agents_positions[x][0] == 0 :
                if self.agents_positions[x][1] % 2:
                   self.action_masks[x][1:4] = 0
                else:
                   self.action_masks[x][2] = 0 
            # if unit is on self.x_max th position it cannot right anymore, mask actions 3,4
            if self.agents_positions[x][1] == (self.width-1) :
                self.action_masks[x][3:5] = 0
            # if unit is on 0th x position it cannot left anymore, mask actions 1,6
            elif self.agents_positions[x][1] == 0 :
                self.action_masks[x][1] = 0
                self.action_masks[x][6] = 0            
            my_state = (*list(my_pos), self.loads[x], *rel_dists, *res_dists, *agent_surround)
            self.obs_dict[x]['observations'] = np.array(my_state, dtype=np.int16)
        # state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr)
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
        for x in self.obs_dict:
            self.obs_dict[x]['action_mask'] = self.action_masks[x]
        self.old_my_units = copy.copy(my_units)
        return self.obs_dict, (x_max, y_max, my_units, enemy_units, self.resources, my_base,enemy_base)
    
    # @staticmethod
    def unit_dicts(self, obs, allies, enemies,  team):
        """This method creates unit dicts to be used in nearest enemy locs."""
        #from the state(obs), following base and dead parameters comes as they are part of a unit. Resources are added just in case.
        unitTagToString = {1: "Truck",2: "LightTank",3: "HeavyTank",4: "Drone",6: "Base",8: "Dead",9: "Resource"}
        lists = [[], []]
        ally_units = obs['units'][team]
        enemy_units = obs['units'][(team+1) % 2]
        
        #creates a list consisting unit types of both sides.
        units_types = [[],[]]
        for i,x in enumerate(allies):
            units_types[0].append(ally_units[x[0]][x[1]])
        for x in enemies:
            units_types[1].append(enemy_units[x[0]][x[1]])
        # units_types = [[ally_units[ally[0], ally[1]] for ally in allies], [enemy_units[enemy[0], enemy[1]] for enemy in enemies]]
        #creates a list consisting unit locations of both sides.
        unit_locations = [allies, enemies]
        #creates a dict for each side consisting unit type tags and unit locations.
        try:
            for i,x in enumerate(unit_locations[0]):
                unit = {
                    "tag" : unitTagToString[units_types[0][i]],
                    "location" : tuple(x)
                    }
                lists[0].append(unit)
        except:
            print('al kirdin kirdin')
            # print(allies)
            # print(units_types)
            # print(unit_locations)
            # print(ally_units)
            # print(enemy_units)
            
        for i in range(len(units_types[1])):
            unit = {
                "tag" : unitTagToString[units_types[1][i]],
                "location" : tuple(unit_locations[1][i])
                }
            lists[1].append(unit)
        return lists[0], lists[1] 
    
    # @staticmethod
    def enemy_unit_dicts(self, new_state, team):
        """This method creates unit dicts to be used in nearest enemy locs."""
        #from the old and new state(old_state, new_state), following base and dead parameters comes as they are part of a unit. Resources are added just in case.
        unitTagToString = {1: "Truck",2: "LightTank",3: "HeavyTank",4: "Drone",6: "Base",8: "Dead",9: "Resource"}
        old_enemy_units = self.nec_obs['units'][(team+1) % 2]
        old_enemy_loc = enemy_locs(self.nec_obs, team)        
        new_enemy_units = new_state['units'][(team+1) % 2]
        new_enemy_loc = enemy_locs(new_state, team)   
        
        #creates a list consisting unit types of both states.
        old_units_types = []
        for x in old_enemy_loc:
            old_units_types.append(old_enemy_units[x[0]][x[1]])
        new_units_types = []
        for x in new_enemy_loc:
            new_units_types.append(new_enemy_units[x[0]][x[1]])

        #creates a dict for each state consisting unit type tags and unit locations.
        old_detail_list = []
        for i,x in enumerate(old_enemy_loc):
            unit = {
                "tag" : unitTagToString[old_units_types[i]],
                "location" : tuple(x)
                }
            old_detail_list.append(unit)
        new_detail_list = []
        for i,x in enumerate(new_enemy_loc):
            unit = {
                "tag" : unitTagToString[new_units_types[i]],
                "location" : tuple(x)
                }
            new_detail_list.append(unit)
            
        return old_detail_list, new_detail_list
    
    def nearest_enemy_details(self, allies, enemies):
            nearest_enemy_detail = []
            for ally in allies:
                # if the ally unit is a truck, append none to nearest enemy list since it is not a fire element and continue to new ally unit.
                if ally["tag"] == "Truck":
                    nearest_enemy_detail.append(None)
                    continue
                if len(enemies) == 0 or len(enemies) < 0:
                    break
                nearest_enemy_detail.append(nearest_enemy_selective(ally, enemies))
            return nearest_enemy_detail
    
    def nearest_enemy_list(self, nearest_enemy_dict):
        nearest_enemy_locs = []
        for n_enemy in nearest_enemy_dict:
            if n_enemy is None:
                nearest_enemy_locs.append(np.asarray([3, 0]))
                continue
            nearest_enemy_locs.append(np.asarray(list(n_enemy["location"])))
        return nearest_enemy_locs
    
    def terrain_locs(self):
        terrain_type = {'d' : 1, 'w' : 2, 'm' : 3}
        if not self.configs['map'].get('terrain'):
            return
        terrain = self.configs['map']['terrain']
        x_max, y_max = self.configs['map']['x'], self.configs['map']['y']
        ter_locs = {}
        for i in range(y_max):
            for j in range(x_max):
               if terrain[i][j] == 'd' or terrain[i][j] == 'w' or terrain[i][j] == 'm':
                    ter_locs[i*self.width+j]=terrain_type[terrain[i][j]]
        return ter_locs
    
    def apply_action(self, action, raw_state, team):
        # this function takes the output of the model and converts it into a reasonable output for the game to play
        blue_score = raw_state["score"][0]
        red_score = raw_state["score"][1]
        # this is specific order as in self.agents
        movement = action[0:7]
        # movement = movement.tolist()
        # target = action[7:14]
        # train = action[14]
        
        # target = []

        enemy_order = []

        # here it changes the units order by creating a set but as long as it keeps consistent no problem
        # but action list is handed by the model by the name of the single agents
        # we have to keep track or we can take directly agents position
        
        allies_ = ally_locs(raw_state, team)
        if not self.firstShot:
            allies__ = ally_locs(self.old_raw_state, team)
        self.firstShot = False
        # this is updated in _decode_state after each game step
        allies = self.agents_positions.values()
        
        counter = 0
        for agent11 in allies_:
            for uni11 in allies:
                if agent11 == uni11:
                    counter +=1
        if counter != len(allies):
            print('Hay amk')
        
        # but how to check if our agent has been killed
        # or a new unit has been created (maybe we can use self.train)
        # but it has to be controlled immediately after game step
        # not here, in _decode_state
        enemies = enemy_locs(raw_state, team)
        my_unit_dict, enemy_unit_dict = self.unit_dicts(raw_state, allies, enemies, team)  
              
        nearest_enemy_dict = self.nearest_enemy_details(my_unit_dict, enemy_unit_dict)
        nearest_enemy_locs = self.nearest_enemy_list(nearest_enemy_dict)
        
        # required for _decode state to decide kill reward
        self.nearest_enemy_locs = []
        self.nearest_enemy_locs = copy.copy(nearest_enemy_locs)
        
        if 0 > len(allies):
            print("Neden negatif adamların var ?")
            raise ValueError
        elif 0 == len(allies):
            locations = []
            movement = []
            target = []
            return [locations, movement, target, self.train]
        elif 0 < len(allies) <= 7:
            ally_count = len(allies)
            locations = allies

            if len(enemies) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)

            while len(enemy_order) > ally_count:
                enemy_order.pop()
            while len(movement) > ally_count:
                # extracting the unused movement parameters
                # there are seven values by default
                # these actions have to be masked
                movement.pop()

        elif len(allies) > 7:
            ally_count = 7
            locations = allies
            if len(enemies) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)
            
            ##added by luchy:due to creating nearest enemy locs for each ally, if number of allies are over 7, only 7 targets must be defined.
            enemy_order = enemy_order[:7]
            
            while len(locations) > 7:
                locations = list(locations)[:7]

        # bu nedir, manuel trucklara 0 atama, yanlis
        # movement = multi_forced_anchor(movement, raw_state, team)

        if len(locations) > 0:
            locations = list(map(list, locations))

        locations = list(map(tuple, locations))
        enemy_order = list(map(tuple, enemy_order))

        # TODO : delete this
        self.old_raw_state = raw_state
        
        number_of_tanks, number_of_enemy_tanks, number_of_uavs, number_of_enemy_uavs, number_of_trucks, number_of_enemy_trucks = 0, 0, 0, 0, 0, 0
        # if hasattr(self, 'my_units'): # it is undefined on the first loop
        for x in my_unit_dict:
            if x["tag"] == "HeavyTank" or x["tag"] == "LightTank":
                number_of_tanks+=1
            elif x["tag"] == "Drone":
                number_of_uavs+=1
            elif x["tag"] == "Truck":
                number_of_trucks+=1
        for x in enemy_unit_dict:
            if x["tag"] == "HeavyTank" or x["tag"] == "LightTank":
                number_of_enemy_tanks+=1
            elif x["tag"] == "Drone":
                number_of_enemy_uavs+=1
            elif x["tag"] == "Truck":
                number_of_enemy_trucks+=1
        
        number_of_our_military = number_of_tanks+number_of_uavs
        number_of_enemy_military =number_of_enemy_tanks+number_of_enemy_uavs
        
        train_truck = False
        train_military = False
        no_train = False
        # priority = 1--> truck, 2--> military
        priority = 0
        self.train = 0
        if blue_score > 0 and raw_state["turn"] > 3:
            if number_of_trucks<1:
                train_truck = True
                priority = 1
            if number_of_our_military<number_of_enemy_military:
                train_military = True
                if priority == 0:
                    priority = 2
            if number_of_trucks < number_of_enemy_trucks:
                train_truck = True
            if raw_state["turn"] / raw_state["max_turn"] > 0.9: 
                no_train = True  
            if raw_state["turn"] > 1 and len(self.resources) / self.init_resource_num < 0.05 and blue_score > red_score+3:
                train_truck = False
            # decide train type.
            if not no_train:
                if priority == 1 and train_truck:
                    self.train = 1
                elif priority == 2:
                    self.train = random.randint(2,3)
                elif train_truck:
                    self.train = 1
                elif train_military:
                    self.train = random.randint(2,4)
        else:
            self.train = 0
        
        # TODO delete this
        # for debug purposes
        # self.train = 1
        
        # if there is a unit 
        # cancel train action
        # this is actually not true since the unit on the base can move into another tile
        # and another unit can move into the base tile
        # we should check actually next positions
        # or should we?
        # we can easily check new units and check if our train action has been implemented
        # unit_on_base = False
        # TODO FIND A BETTER SOLUTION
        # for x in self.agents:
        #     if self.agents_positions[x] == self.my_base:
        #         unit_on_base = True
        #         break
        # if unit_on_base:
        #     self.train = 0

        '''
        # if blue_score > 0:
        #     if raw_state["turn"] / raw_state["max_turn"] > 0.9: 
        #         self.train = 0 
        #     elif raw_state["turn"] > 1 and len(self.resources) / self.init_resource_num < 0.05 and blue_score > red_score+3:
        #         self.train = 0
        #     elif number_of_trucks<1:
        #         self.train = 1
        #     elif blue_score>red_score+1:
        #         if number_of_trucks<2:
        #             self.train = 1
        #         elif number_of_tanks<1:
        #             self.train = 2
        #         elif number_of_uavs<1:
        #             self.train = 4
        #     elif blue_score>red_score+2:
        #         if number_of_trucks<2:
        #             self.train = 1
        #         elif number_of_tanks<1:
        #             self.train = 3
        #         elif number_of_uavs<1:
        #             self.train = 4
        #         elif number_of_our_military<number_of_enemy_military:
        #             self.train = random.randint(2,4)
        #     elif blue_score < red_score and blue_score > 0:
        #         if number_of_trucks<1 or (number_of_trucks<2 and len(self.resources) / self.init_resource_num) > 0.6:
        #             self.train = 1
        #         elif number_of_trucks>1 and number_of_tanks<1:
        #             self.train = random.randint(2,3)
        #         elif number_of_trucks>1 and number_of_uavs<1:
        #             self.train = 4
        # else:
        #     self.train = 0
        '''   

        # this has to be returned in this order according to challenge rules
        return [locations, movement, enemy_order, self.train]
    def step(self, action_dict):
        # wait a little bit
        # self.action_mask = np.ones(49,dtype=np.int8)
        
        # dont forget to reset reward
        for x in self.agents:
            self.rewards[x] = 0
        
        self.current_action = action_dict

        harvest_reward, kill_reward, martyr_reward = 0, 0, 0

        # self.env.step(action[self.env.agent_selection])

        # we are expectin an action dictionary of agents
        action = []
        for x in self.agents:
            action.append(action_dict[x])
        # action = np.array([x for x in action_dict.values()])
        action = self.apply_action(action, self.nec_obs, self.team)
        next_state, _, done =  self.game.step(action)

        # we have to update our agents and agents_locations list
        # immediately
        obs_d, next_info = self._decode_state(next_state)
        # _, info = self._decode_state(self.nec_obs)
        # here we will convert the action space
        # into the game inputs as location, movement, target and train
        # game step returns next_state,reward,done

        # obs_d = {} # we got these from _decode_state
        rew_d = self.rewards

        # get rewards
        # we managed this in decode state
        
        if done: #this comes from game step
            for x in self.agents:
                self.dones[x] = True
            self.dones['__all__'] = True

        done_d = self.dones
        

        # TODO: wildcard
        # self.infos comes empty check this
        for x in obs_d:
            self.infos[x] = {}
        info_d = copy.copy(self.infos)
        self.nec_obs = next_state

        return obs_d, rew_d, done_d, info_d


