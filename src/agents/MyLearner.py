import copy
from os import kill
import random
from agents.BaseLearningGym import BaseLearningAgentGym
import gym
from gym import spaces
import numpy as np
import yaml
from game import Game
from utilities import multi_forced_anchor, necessary_obs, decode_location, multi_reward_shape, enemy_locs, ally_locs, getDistance, nearest_enemy, truck_locs


def read_hypers():
    with open(f"/workspaces/Suru2022/data/config/RiskyValleyNoTerrain.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict


class MyLearner(BaseLearningAgentGym):

    tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

    def __init__(self, args, agents, team=0):
        super().__init__() 
        self.configs = read_hypers()
        # self.resetMap =copy.deepcopy(self.configs)
        # this wont make any difference here
        # configs['blue']['base']['x'] = 3
        self.game = Game(args, agents)
        # this works, but find where it is reset
        # self.game.config['blue']['base']['x'] = 3
        # call this in reset function
        # self.manipulateMap(self.game.config)
        self.mapChangeFrequency = 1000
        # original map size
        self.gameAreaX = 6
        self.gameAreaY = 4
        self.train = 0

        self.team = team
        self.enemy_team = 1
        
        self.height = self.configs['map']['y']
        self.width = self.configs['map']['x']
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None

        # define the action mask
        self.action_mask = np.ones(103,dtype=np.int8)

        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(24*18*10+4,),
            dtype=np.int16
        )
        # self.action_space = self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5])
        # exclude the last action and manage it in this script, check simpleagent for it
        # self.action_space = self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
        # self.action_space = self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, 7 ])
        self.observation_space = spaces.Dict (
            {
            "observations": spaces.Box(
            low=-2,
            high=401,
            shape=(24*18*10+4,),
            # shape=(6*4*10+4,),
            dtype=np.int16
        ),
            # "action_mask" : spaces.Box(0.0, 1.0, shape=self.action_space.shape) }
            # "action_mask" : spaces.Box(0, 1, shape=(103,),dtype=np.int8) }
            "action_mask" : spaces.Box(0, 1, shape=(49,),dtype=np.int8) }
        )
        self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, 7])
        # TODO : check this hele
        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

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
        if(episode%self.mapChangeFrequency==0):
        # if(False):
            # print(episode)
            self.resetPosition(mapDict)
            xOffSet = random.randint(0,self.width-self.gameAreaX)
            yOffSet = random.randint(0,self.height-self.gameAreaY)
            self.addOffSet(mapDict['blue']['base'],xOffSet, yOffSet)
            self.addOffSet(mapDict['red']['base'],xOffSet, yOffSet)
            for x in mapDict['blue']['units']:
                self.addOffSet(x,xOffSet, yOffSet)
            for x in mapDict['red']['units']:
                self.addOffSet(x,xOffSet, yOffSet)
        
        # find out already occupied tiles
        occupiedTiles = {self.getCoordinate(mapDict['blue']['base']), self.getCoordinate(mapDict['red']['base'])}
        for x in mapDict['blue']['units']:
            occupiedTiles.add(self.getCoordinate(x))
        for x in mapDict['red']['units']:
            occupiedTiles.add(self.getCoordinate(x))

        # randomize resource positions
        for x in mapDict['resources']:
            a = random.randint(0, self.gameAreaX-1)+xOffSet
            b = random.randint(0, self.gameAreaY-1)+yOffSet
            while self.getCoordinate({'x':a,'y':b}) in occupiedTiles:
                a = random.randint(0, self.gameAreaX-1)+xOffSet
                b = random.randint(0, self.gameAreaY-1)+yOffSet
            occupiedTiles.add(self.getCoordinate({'x':a,'y':b}))
            x['x'] = a
            x['y'] = b

    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        # print("setup")

    def reset(self):
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0

        # try that
        # print(self.episodes)

        # change it on every episode
        # self.manipulateMap(self.game.config, self.episodes)
        state = self.game.reset()
        self.nec_obs = state
        return self.observation_space.sample()

    @staticmethod
    def  _decode_state(obs, team, enemy_team):
        turn = obs['turn'] # 1
        max_turn = obs['max_turn'] # 1
        units = obs['units'] # 2x7x15
        hps = obs['hps'] # 2x7x15
        bases = obs['bases'] # 2x7x15
        score = obs['score'] # 2
        res = obs['resources'] # 7x15
        load = obs['loads'] # 2x7x15
        terrain = obs["terrain"] # 7x15 
        y_max, x_max = res.shape
        my_units = []
        enemy_units = []
        resources = []
        for i in range(y_max):
            for j in range(x_max):
                if units[team][i][j]<6 and units[team][i][j] != 0:
                    my_units.append(
                    {   
                        'unit': units[team][i][j],
                        'tag': MyLearner.tagToString[units[team][i][j]],
                        'hp': hps[team][i][j],
                        'location': (i,j),
                        'load': load[team][i][j]
                    }
                    )
                if units[enemy_team][i][j]<6 and units[enemy_team][i][j] != 0:
                    enemy_units.append(
                    {   
                        'unit': units[enemy_team][i][j],
                        'tag': MyLearner.tagToString[units[enemy_team][i][j]],
                        'hp': hps[enemy_team][i][j],
                        'location': (i,j),
                        'load': load[enemy_team][i][j]
                    }
                    )
                if res[i][j]==1:
                    resources.append((i,j))
                if bases[team][i][j]:
                    my_base = (i,j)
                if bases[enemy_team][i][j]:
                    enemy_base = (i,j)
        
        # print(my_units)
        unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        ress = [*res.reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        terr = [*terrain.reshape(-1).tolist()]
        
        state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr)

        return np.array(state, dtype=np.int16), (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)

    @staticmethod
    def just_decode_state(obs, team, enemy_team):
        state, _ = MyLearner._decode_state(obs, team, enemy_team)
        return state

    def decode_state(self, obs):
        state, info = self._decode_state(obs, self.team, self.enemy_team)
        self.x_max, self.y_max, self.my_units, self.enemy_units, self.resources, self.my_base, self.enemy_base = info
        return state

    @staticmethod
    def just_decode_state_(obs, team, enemy_team):
        state, info = MyLearner._decode_state(obs, team, enemy_team)
        return state, info
    
    def take_action(self, action):
        return self.just_take_action(action, self.nec_obs, self.team, self.train) 

    @staticmethod
    def unit_types(obs, allies, enemies,  team):
        ally_units = obs['units'][team]
        enemy_units = obs['units'][(team+1) % 2]
        return [ally_units[ally[0], ally[1]] for ally in allies], [enemy_units[enemy[0], enemy[1]] for enemy in enemies]
    
    @staticmethod
    def just_take_action(action, raw_state, team, train):
        # this function takes the output of the model
        # and converts it into a reasonable output for
        # the game to play
        movement = action[0:7]
        movement = movement.tolist()
        # target = action[7:14]
        # train = action[14]
        
        # target = []

        enemy_order = []

        allies = ally_locs(raw_state, team)
        enemies = enemy_locs(raw_state, team)
        my_unit_types, _ = MyLearner.unit_types(raw_state, allies, enemies, team)    
        # nearest_enemy_locs = []
        # distances = []
        # for ally in allies:
        #     if len(enemies) == 0 or len(enemies) < 0:
        #         break
        #     nearest_e, distance = nearest_enemy(ally, enemies)    
        #     nearest_enemy_locs.append(nearest_e)
        #     distances.append(distance)
        nearest_enemy_locs = []
        for ally in allies:
            if len(enemies) == 0 or len(enemies) < 0:
                break
            nearest_enemy_locs.append(nearest_enemy(ally, enemies))
            
        if 0 > len(allies):
            print("Neden negatif adamların var ?")
            raise ValueError
        elif 0 == len(allies):
            locations = []
            movement = []
            target = []
            return [locations, movement, target, train]
        elif 0 < len(allies) <= 7:
            ally_count = len(allies)
            locations = allies

            # counter = 0
            # for j in target: 
            #     if len(enemies) == 0:
            #         # yok artik alum
            #         enemy_order = [[3, 0] for i in range(ally_count)]
            #         continue
            #     k = j % len(enemies)
            #     if counter == ally_count:
            #         break
            #     if len(enemies) <= 0:
            #         break
            #     enemy_order.append(enemies[k].tolist())
            #     counter += 1

            ##added by luchy: this part creates a list of closest enemy order. If num of enemies == 0 creates a dummy fire point for each ally.
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
        
        # mask out the unused part of the action space
        # This have to be set at the beginning
        # or no
        # This value changes throughout the sim
        # self.action_mask[]

        
        elif len(allies) > 7:
            ally_count = 7
            locations = allies

            # counter = 0
            # for j in target:
            #     if len(enemies) == 0:
            #         # bu ne oluyor press tv
            #         enemy_order = [[3, 0] for i in range(ally_count)]
            #         continue
            #     k = j % len(enemies)
            #     if counter == ally_count:
            #         break
            #     if len(enemies) <= 0:
            #         break
            #     enemy_order.append(enemies[k].tolist())
            #     counter += 1
            ##added by luchy:
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
        
        # boyle bisi olabilir mi ya
        # locations'dan biri, bir düşmana 2 adımda veya daha yakınsa dur (movement=0) ve ona ateş et (target = arg.min(distances))
        # for i in range(len(locations)):
        #     for k in range(len(enemy_order)):
        #         if getDistance(locations[i], enemy_order[k]) <= 3:
        #             movement[i] = 0
        #             enemy_order[i] = enemy_order[k]

        # also a model manipulation, prevents model learning that actually
        ##added by luchy:by this if the distance between ally and enemy is less than 3 then movement will be 0 as a preparation to shoot.
        # for i in range(len(locations)):
        #     if getDistance(locations[i], enemy_order[i]) <= 3:
        #         movement[i] = 0

        locations = list(map(tuple, locations))
        enemy_order = list(map(tuple, enemy_order))

        # this has to be returned in this order according to challenge rules
        # return [locations, movement, enemy_order, train], distances
        return [locations, movement, enemy_order, train]
    def step(self, action):
        self.action_mask = np.ones(49,dtype=np.int8)

        harvest_reward = 0
        kill_reward = 0
        martyr_reward = 0
        action = self.take_action(action)

        next_state, _, done =  self.game.step(action)

        next_state_obs, next_info = self.just_decode_state_(next_state,self.team,self.enemy_team)
        # next_state_obs = self.decode_state(next_state)

        # ##added by luchy:for following counter required
        #         _, info = MyLearner.just_decode_state_(self.nec_obs, self.team, self.enemy_team)
        #         self.x_max, self.y_max, self.my_units, self.enemy_units, self.resources, self.my_base, self.enemy_base = info
        
        harvest_reward, enemy_count, ally_count = multi_reward_shape(self.nec_obs, self.team, action)
        if enemy_count < self.previous_enemy_count:
            kill_reward = (self.previous_enemy_count - enemy_count) * 5
        if ally_count < self.previous_ally_count:
            martyr_reward = (self.previous_ally_count - ally_count) * 5
        # only reward goes for collecting gold
        # reward = harvest_reward + kill_reward - martyr_reward
        reward = harvest_reward

        # reward = harvest_reward
        # reward = 0

        # consider givin reward only at episode end
        blue_score = next_state['score'][0]
        red_score = next_state['score'][1]

        # if done:
        #     if blue_score>red_score:
        #         reward = 1
        #     else:
        #         reward = -1

        # we have to discourage training action
        # like in simple agent, our agent has to keep its resources
        # the number of the entities can be compared. 
        # We actually need enough trucks to exploit maps resources
        # at the same time we have to keep military entities for defence        
        # 
        
        # entity_train = action[-1] # 0 for none 1 for truck up to 4 for other attack units
        
        # if you have 2 or less gold dont train anything other than truck
        # if blue_scsore < 3 and entity_train > 1:
        #     reward = -100
        #     done = True
        # this didnt work

        # check unit numbers - make this a seperated function
        # Train: 0-4 arası tam sayı (integer, kısaca int). 0 ünite yapmamayı, 1-4 ise sırasıyla kamyon, hafif tank,
        # ağır tank ve İHA yapmayı ifade etmektedir
        number_of_tanks, number_of_enemy_tanks, number_of_uavs, number_of_enemy_uavs, number_of_trucks, number_of_enemy_trucks = 0, 0, 0, 0, 0, 0
        if hasattr(self, 'my_units'): # it is undefined on the first loop
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

            # assuming that self.my_units is in the exact order as in locations list
            # here we can mask out unused action spaces for non existence units
            # this didnt work
            # rather than masking the whole remnant action space
            # we can define specific non-playable actions i think
            # check this
            # self.action_mask[len(self.my_units)*7:49] = 0
            # rather than using self.my_units we should check unit number in the next state
            # self.action_mask[len(next_info[2])*7:49] = 0
            # self.action_mask[49+len(self.my_units)*7:98] =0
        
        # we are no longer dependent on the self.my_units
        # so we took this from the upper condition
        # next_info[2] refers to our units
        self.action_mask[len(next_info[2])*7:49] = 0
        # entity_train = action[-1]

        # mask actions other than load or unload (0) if the truck is on a resource or the base
        resource_loc = np.argwhere(next_state['resources'] == 1)
        units_in_next_state = next_info[2]
        base = next_info[5]
        # trucks_loc = truck_locs(next_state,self.team)

        # we assume that all unit lists are ordered by their coordinate
        # next_info[2] is our units in the next state
        for i,unit in enumerate(units_in_next_state):
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
            for reso in resource_loc:            
                # if there is resource on the next location of the truck
                if (reso == unit['location']).all():
                        # find the index of the truck in the unit list
                        self.action_mask[i*7]=1
                        # mask actions other than 0
                        self.action_mask[i*7+1:i*7+7]=0

        number_of_our_military = number_of_tanks+number_of_enemy_uavs
        number_of_enemy_military =number_of_enemy_tanks+number_of_enemy_uavs

        # early_termination = True
        # # if there are resources to spend
        # if blue_score>0:

        #     # catch and beat the enemy military numbers
        #     if number_of_enemy_military>=number_of_our_military and entity_train>1:
        #         reward+=10
        #         early_termination = False

        #     # if there is no truck, train truck
        #     # what about reward scale?
        #     if number_of_trucks == 0 or number_of_enemy_trucks>=number_of_trucks:
        #         if entity_train == 1:
        #             reward+=10
        #             early_termination = False

        #     # reason about the resources that we already have
        #     # make it hard for the model to train anything but necessary
        #     if entity_train > 0:
        #         reward-=10
        #         if early_termination and blue_score<3:
        #             done = True
        #             # instead of to early terminate mask the train action
        #             # set the last 4 element to zero
        #             # how can i be sure about this
        #             # self.action_mask[-4:]=0

        # copied from simple agent
        # this is rule based unit creation logic
        if blue_score>red_score+2:
            if number_of_trucks<2:
                self.train = 1
            elif number_of_tanks<1:
                self.train = random.randint(2,3)
            elif number_of_uavs<1:
                self.train = 4
            elif number_of_our_military<number_of_enemy_military:
                self.train = random.randint(2,4)
        elif number_of_trucks<1:
            self.train = 1
        elif blue_score+2<red_score and len(self.my_units)<len(self.enemy_units)*2:
            self.train = random.randint(2,4)


        self.previous_enemy_count = enemy_count
        self.previous_ally_count = ally_count
        info = {}
        self.steps += 1
        self.reward += reward

        self.nec_obs = next_state
        # return self.decode_state(next_state), reward, done, info
        # return{ "observations":self.decode_state(next_state),"action_mask":self.action_mask}, reward, done, info
        return {"observations":next_state_obs,"action_mask":self.action_mask}, reward, done, info

    def render(self,):
        return None

    def close(self,):
        return None