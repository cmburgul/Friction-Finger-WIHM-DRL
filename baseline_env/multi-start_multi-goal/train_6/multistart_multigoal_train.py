import numpy as np
import pyglet
import random
import time
import matplotlib.pyplot as plt
from math import radians, degrees
from collections import deque

import gym
from gym import spaces, error
from gym.utils import seeding, closer
from utils import *

# from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.deepq.policies import FeedForwardPolicy

env_closer = closer.Closer()
STATE_SIZE = 7
ACTION_SIZE = 5
low = np.array([radians(0), radians(0), 0.5, 0.5, -2.0, -2.0, 0])
high = np.array([radians(360), radians(360), 2.5, 2.5, 2.0, 2.0, 1])
# # Full Workspace 
MAX_OBJ_LIMIT = 105 # in mm
MIN_OBJ_LIMIT = 25  # in mm
Δθ = 1 # Actuation magnitude

# tl : theta_left
# dl : distance from object base to left finger base
# tr : theta_right
# dr : distance from object base to right finger base

class FFEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    
    viewer = None
    dt = 0.01    # refresh rate

    state_size = STATE_SIZE
    action_size = ACTION_SIZE
    center_coord = np.array([100, 0])
    low = np.array([radians(0), radians(0), 0.5, 0.5, -2.0, -2.0, 0])
    high = np.array([radians(180), radians(180), 2.5, 2.5, 2.0, 2.0, 1])

    w0 = 25  # Object width
    wp = 50  # 
    fw = 18  # Finger width

    def __init__(self):
        super(FFEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()

        self.ff_info = np.zeros(2, dtype=[('d', np.float32), ('t', np.float32), ('a', np.int)])
        self.goal = {'x': 0., 'y': 0., 'w':self.w0} # Goal Position of the Object 
        self.goal['x'], self.goal['y'] = self.get_goal_point()
        # self.goal['x'], self.goal['y'] = 388, 228
        
        # Intialising with sliding on left finger 
        #self.ff_info['t'][1] = radians(50)      # Initialising tr in deg
        #self.ff_info['d'][1] = 80               # Initialising dr in mm
        #self.ff_info['t'][0], self.ff_info['d'][0] = self.calc_left_config(self.ff_info['t'][1], self.ff_info['d'][1])
        
        # Initialising with sliding on right finger
        self.ff_info['t'][0] = radians(90)        # Initialising tl in deg
        self.ff_info['d'][0] = 35                # Initialising dl in mm
        self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0])

        self.obj_pos = {'x':0., 'y':0.} # Object Position
        self.on_goal = 0

        self.ff_info['a'] = 0

        self.time_step_i = 0
        self.time_steps = 1000
        self.episode = 1

        self.log = logger()
        self.log.add_log('scores')
        self.log.add_log('avg_loss')
        self.scores = []                        # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.score = 0 
        self.done = False
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        done = False
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # action : Action is discrete
        # 0 : a = [0 0]
        # 1 : a = [+Δθ 0]   Moving Left Finger in CCW -- Sliding on Right Finger Up  
        # 2 : a = [-Δθ 0]   Moving Left Finger in CW  -- Sliding on Right Finger Down
        # 3 : a = [0 +Δθ]   Moving Right Finger in CCW -- Sliding on Left Finger Down
        # 4 : a = [0 -Δθ]   Moving Right Finger in CW  -- Sliding on Left Finger Up
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        #print("action from agent : ", action)
        
        #print("Action choosen is Do Nothing")
        if (action == 0):
            self.ff_info['t'][0] = self.ff_info['t'][0] # Hovering Object at a place (or) taking a pause 

        if (action == 1): # Action is Sliding on Right Finger Up  
            # 1 : a = [+Δθ_l]   Moving Left Finger in CCW -- Sliding on Right Finger Up  
            # Give θ_l and d_l values to get θ_r and d_r values
            # Add the actuation of +Δθ in θ_l 

            #print("Action choosen is Sliding on Right Finger Up")
            
            # Creating dummy variables t0, d0, t1, d1
            # Copying values of ff_info['t'][0], ff_info['t'][1] to dummy variables t0, d0
            self.t0 = self.ff_info['t'][0]
            self.d0 = self.ff_info['d'][0]

            # Adding the action delta theta to theta_left or t_l and sliding on right finger
            self.t0 += Δθ * self.dt 

            # Actuation flag 
            action_flag = True

            # Constraining the left finger to lesser than 140 deg
            if ( self.t0 >= radians(140) ):
                action_flag = False
                #print("Crossing limits t0 >= 140")
            elif (self.d0 >= MAX_OBJ_LIMIT ):
                action_flag = False
                #print("Crossing limits d0 >= 105")
            elif (self.d0 <= MIN_OBJ_LIMIT):
                action_flag = False
                #print("Crossing limits d0 <= 25")

            # Getting tr, dr by giving tl, d1
            self.t1, self.d1 = self.calc_right_config(self.t0, self.d0)

            # Constraining t1, d1 limits
            if ( self.d1 >= MAX_OBJ_LIMIT ):
                action_flag = False
                #print("Crossing limits d1 >= 105")
            elif (self.d1 <= MIN_OBJ_LIMIT):
                action_flag = False
                #print("Crossing limits d1 <= 25")
            elif ( self.t1 <= radians(40) or self.t1 >= radians(152) ): # greater than 140 deg
                action_flag = False
                #print("Crossing limits t1 <= 40")

            #elif (action_flag == False):
                #print(" Action not taken") 

            if (action_flag == True):
                #print("Action of Sliding on Right Finger Up is taken")
                # For sliding right take action for tl, dl and get tr, dr
                self.ff_info['t'][0] += Δθ * self.dt  
                #self.ff_info['t'][1] %= np.pi * 2 # normalize ! Why ? Need to know         
                
                # Getting tr, dr by giving tl, dl
                self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0]) 

        elif (action == 2): # Action is Sliding on Right Finger Down
            # 2 : a = [-Δθ_l]   Moving Left Finger in CW  -- Sliding on Right Finger Down
            # Give θ_l and d_l values to get θ_r and d_r values
            # Add the actuation of -Δθ in θ_l
            
            #print("Action choosen is Sliding on Right Finger Down")
            
            # Creating dummy variables t0, d0, t1, d1
            # Copying values of ff_info['t'][0], ff_info['t'][1] to dummy variables t0, d0
            self.t0 = self.ff_info['t'][0]
            self.d0 = self.ff_info['d'][0]

            # Adding the action delta theta to theta_left or t_l and sliding on right finger
            self.t0 -= Δθ * self.dt 

            # Actuation flag 
            action_flag = True

            # Constraining the left finger to lesser than 140 deg
            if ( self.t0 >= radians(140) ):
                action_flag = False
                #print("Crossing limits t0 >= 140")
            elif (self.d0 >= MAX_OBJ_LIMIT ):
                action_flag = False
                #print("Crossing limits d0 >= 105")
            elif (self.d0 <= MIN_OBJ_LIMIT):
                action_flag = False
                #print("Crossing limits d0 <= 25")

            # Getting tr, dr by giving tl, d1
            self.t1, self.d1 = self.calc_right_config(self.t0, self.d0)

            # Constraining t1, d1 limits
            if ( self.d1 >= MAX_OBJ_LIMIT ):
                action_flag = False
                #print("Crossing limits d1 >= 105")
            elif (self.d1 <= MIN_OBJ_LIMIT):
                action_flag = False
                #print("Crossing limits d1 <= 25")
            elif ( self.t1 <= radians(40) or self.t1 >= radians(152) ): # greater than 140 deg
                action_flag = False
                #print("Crossing limits t1 <= 40")

            #elif (action_flag == False):
                #print(" Action not taken") 

            if (action_flag == True):
                #print("Action of Sliding on Right Finger Up is taken")
                # For sliding right take action for tl, dl and get tr, dr
                self.ff_info['t'][0] -= Δθ * self.dt  
                
                #self.ff_info['t'][1] %= np.pi * 2 # normalize ! Why ? Need to know         
                
                # Getting tr, dr by giving tl, dl
                self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0]) 

        elif (action == 3): # Action is Sliding on Left Finger Down
            # 3 : a = [+Δθ_r]   Moving Right Finger in CCW -- Sliding on Left Finger Down
            # Give θ_r and d_r values to get θ_l and d_l values
            # Add the actuation of +Δθ in θ_r

            #print("Action choosen is Sliding on Left Finger Down")
            
            # Creating dummy variables t0, d0, t1, d1
            self.t1 = self.ff_info['t'][1]
            self.d1 = self.ff_info['d'][1]
          
            self.t1 += Δθ * self.dt # Adding the action delta theta to theta_right or θ_r

            # Actuation flag 
            action_flag = True

            # Constraining the right finger 
            if (self.t1 < radians(40)):   # limit it to greater than 40 deg
                action_flag = False
                #print("Crossing limits t1 < 40")
            if (self.d1 >= MAX_OBJ_LIMIT):   # limit it to 105 mm 
                action_flag = False
                #print("Crossing limits d1 > 105")
            if (self.d1 <= MIN_OBJ_LIMIT):    # limit it to 25 mm
                action_flag = False
                #print("Crossing limits d1 < 25")

            # Getting tl, dl by giving tr, dr
            self.t0, self.d0 = self.calc_left_config(self.t1, self.d1) 

            # Constraining tl, dl limits  
            if (self.d0 >= MAX_OBJ_LIMIT): # Maximum limit
                action_flag = False
                #print("Crossing limits d0 > 105")
            if (self.d0 <= MIN_OBJ_LIMIT): # limit it to 25 mm
                action_flag = False
                #print("Crossing limits d0 < 25")
            if (self.t0 >= radians(140)):  # limit t0 at 140 deg
                action_flag = False
                #print("Crossing limits t0 > 140")

            #if (action_flag == False):
                #print(" Action not taken") 
            
            elif (action_flag == True):
                #print("Action of Sliding on Left Finger Down is taken")
                
                self.ff_info['t'][1] += Δθ * self.dt   # Subtracting the action delta theta to theta_right or θ_r
                
                #self.ff_info['t'][1] %= np.pi * 2 # normalize ! Why ? Need to know
                
                # Getting tl, dl by giving tr, dr
                self.ff_info['t'][0], self.ff_info['d'][0] = self.calc_left_config(self.ff_info['t'][1], self.ff_info['d'][1])   

        elif (action == 4): # Action is Sliding on Left Finger Up
            # 4 : a = [-Δθ_r]   Moving Right Finger in CW  -- Sliding on Left Finger Up
            # Give θ_r and d_r values to get θ_l and d_l values
            # Add the actuation of -Δθ in θ_r

 
            #print("Action choosen is Sliding on Left Finger UP")
            
            # Creating dummy variables t0, d0, t1, d1
            self.t1 = self.ff_info['t'][1]
            self.d1 = self.ff_info['d'][1]
          
            self.t1 -= Δθ * self.dt # Adding the action delta theta to theta_right or θ_r

            # Actuation flag 
            action_flag = True

            # Constraining the right finger 
            if (self.t1 < radians(40)):   # limit it to greater than 40 deg
                action_flag = False
                #print("Crossing limits t1 < 40")
            if (self.d1 >= MAX_OBJ_LIMIT):   # limit it to 105 mm 
                action_flag = False
                #print("Crossing limits d1 > 105")
            if (self.d1 <= MIN_OBJ_LIMIT):    # limit it to 25 mm
                action_flag = False
                #print("Crossing limits d1 < 25")

            # Getting tl, dl by giving tr, dr
            self.t0, self.d0 = self.calc_left_config(self.t1, self.d1) 

            # Constraining tl, dl limits  
            if (self.d0 >= MAX_OBJ_LIMIT): # Maximum limit
                action_flag = False
                #print("Crossing limits d0 > 105")
            if (self.d0 <= MIN_OBJ_LIMIT): # limit it to 25 mm
                action_flag = False
                #print("Crossing limits d0 < 25")
            if (self.t0 >= radians(140)):  # limit t0 at 140 deg
                action_flag = False
                #print("Crossing limits t0 > 140")

            #if (action_flag == False):
                #print(" Action not taken") 
            
            elif (action_flag == True):
                #print("Action of Sliding on Left Finger UP is taken")
                
                self.ff_info['t'][1] -= Δθ * self.dt   # Subtracting the action delta theta to theta_right or θ_r
                
                #self.ff_info['t'][1] %= np.pi * 2 # normalize ! Why ? Need to know
                
                # Getting tl, dl by giving tr, dr
                self.ff_info['t'][0], self.ff_info['d'][0] = self.calc_left_config(self.ff_info['t'][1], self.ff_info['d'][1])         

        #print('ff_info : ', self.ff_info)

        # State        
        # Object Position
        # For getting Object Position there is a difference in slide_obj_right and slide_obj_left. 
        # Let's follow a standard of getting Object Position from slide_obj_right
        self.obj_pos['x'], self.obj_pos['y'] = self.get_obj_slide_right(self.ff_info['t'][0], self.ff_info['d'][0])  
        
        #print("step function obj_pos : ", self.obj_pos )

        # Distance from goal to object
        dist_x = (self.obj_pos['x'] - self.goal['x'])
        dist_y = (self.obj_pos['y'] - self.goal['y'])
        orient = (self.ff_info['o'][0] - self.goal['o'])
        
        # done and reward
        #print("dist_x :", dist_x)
        #print("dist_y :", dist_y)
        
        # reward engineering
        reward = -np.sqrt((dist_x**2 + dist_y**2))/200 + orient/180
        #print(reward)
        # Sparse reward
        #reward = -1 # Sparse reward Calculate distance between obj_pos and goal_pos
        
        # Check if object is near to goal in x-axis
        if ( self.goal['x'] - self.goal['w']/2 < self.obj_pos['x'] < self.goal['x'] + self.goal['w']/2 ):
            # Check if object is near to goal in y-axis
            if ( self.goal['y'] - self.goal['w']/2 < self.obj_pos['y'] < self.goal['y'] + self.goal['w']/2 ):
                reward += 0.
                self.on_goal += 1
                if self.on_goal > 5:
                    self.done = True
        else:
            self.on_goal = 0

        # Concatenate and normalize
        next_state = np.concatenate((self.ff_info['t'][0], 
                                     self.ff_info['t'][1], 
                                     self.obj_pos['x']/200, 
                                     self.obj_pos['y']/200, 
                                     dist_x/200, 
                                     dist_y/200, 
                                     [1. if self.on_goal else 0.]), axis=None)

        info = {}

        self.time_step_i = self.time_step_i + 1
        self.score += reward
        #print("Episode : ", self.episode, " time_step_i : ", self.time_step_i)


        if ((self.time_step_i >= self.time_steps) or (self.done == True)):
            self.episode = self.episode + 1
            self.scores_window.append(self.score)
            self.scores.append(self.score)
            self.log.add_item('scores', self.score)
            self.score = 0
            self.reset()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, np.mean(self.scores_window)), end="")
            if self.episode % 10 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, np.mean(self.scores_window)))
            self.time_step_i = 0
        
        return next_state, reward, done, info
    
    def reset(self):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # During start of every episode the agent will reset the envirohnment 
        # 1. Gives a new goal location
        # 2. Gives a initial state of the system 
        #
        # Input : none
        # Output : state
        # State : { theta_l, theta_r, O_x, O_y, (G-O)_x, (G-O)_y, done }
        self.num_time_steps = 0
        self.done = False
        self.ff_info['t'][0] = radians(np.random.randint(90, 110))
        self.ff_info['d'][0] = np.random.randint(55, 75)
        self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0])
        
        # Goal location 
        self.goal['x'], self.goal['y'] = self.get_goal_point() # Multi-Goal RL
        # self.goal['x'], self.goal['y'] = 388, 228
        
        # Object Position
        # There is a difference in getting object position with slide_obj_right and slide_obj_left functions 
        # Let's follow a standard of slide_obj_right
        self.obj_pos['x' ], self.obj_pos['y'] = self.get_obj_slide_right(self.ff_info['t'][0], self.ff_info['d'][0])        
        #obj_pos_slide_left = self.get_obj_slide_left(self.ff_info['t'][1], self.ff_info['d'][1])

        # Distance from goal to object
        dist_x = self.obj_pos['x'] - self.goal['x'] 
        dist_y = self.obj_pos['y'] - self.goal['y']

        self.episode = self.episode + 1
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.log.add_item('scores', self.score)
        self.score = 0
        # self.reset()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, np.mean(self.scores_window)), end="")
        if self.episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, np.mean(self.scores_window)))
        # self.time_step_i = 0

        state = np.concatenate((self.ff_info['t'][0], 
                                self.ff_info['t'][1], 
                                self.obj_pos['x']/200, 
                                self.obj_pos['y']/200, 
                                dist_x/200, 
                                dist_y/200, 
                                [1. if self.on_goal else 0.]), axis=None)        
        return state

    def close(self):
        return
    
    def calc_left_config(self, tr, dr):
        d2v = np.array([dr * np.cos(np.float64(tr)), dr * np.sin(np.float64(tr))])
        w0v = np.array([self.w0 * np.sin(np.float64(tr)), -self.w0 * np.cos(np.float64(tr))])
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * np.sin(np.float64(tr)), -self.fw * np.cos(np.float64(tr))])
        av = d2v - f1v - w0v + wpv
        # Calculate of thetal, dl
        dl = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        tl = np.arctan2(float(av[1]), float(av[0])) + np.arctan2(self.fw, dl)
        return tl, dl

    def calc_right_config(self, tl, dl):
        d1v = np.array([dl * np.cos(tl), dl * np.sin(tl)])
        d1v = d1v.reshape(1,2)[0]
        w0v = np.array([self.w0 * np.sin(tl), -self.w0 * np.cos(tl)])
        w0v = w0v.reshape(1,2)[0]
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * np.sin(tl), -self.fw * np.cos(tl)])
        f1v = f1v.reshape(1,2)[0]
        av = d1v + w0v + f1v - wpv
        # Calculate thetar, dr
        dr = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        tr = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(self.fw, dr)
        return tr, dr

    def sample_action(self):
        # Discrete action space
        list_ = [0, 1, 2, 3, 4]             # Create a list of [0, 1, 2, 3, 4] 
        return random.choice(list_)        # Randomly select from list_        
  
    # Additional function
    def slope(self,x1, y1, x2, y2):
        m = float(float((y2 - y1))/float((x2 - x1)))
        return m

    # Additional function
    def get_goal_point(self):  
        # Defining Points from A - D 
        A_x, A_y = (109.766, 226.160)
        B_x, B_y = (174.045, 149.555)
        C_x, C_y = (347.566, 163.047)
        D_x, D_y = (413.553, 234.954)

        goal_flag = True
        
        while (goal_flag):
            
            section = None
            
            # Randomly selecting X - Co-ordinates within limits
            x_g = np.random.randint(109.766, 413.553, size=1).astype("float64") # Search for reasoning of limits
            # Randomly selecting Y - Co-ordinates within limits
            y_g = np.random.randint(149.5, 234.954, size=1).astype("float64")
            
            # Then sort it based on the section
            if (109.766 < x_g < 174.045):
                section = 1
            elif (174.045 < x_g < 347.566):
                section = 2
            elif (347.566 < x_g < 413.553):
                section = 3    
                
            if (section == 1):
                # Slope of line AD
                m_ad = self.slope(D_x, D_y, A_x, A_y)
                # Slope of point P and point A
                m_ap = self.slope(A_x, A_y, x_g, y_g)
                # Slope of point A and point B
                m_ab = self.slope(A_x, A_y, B_x, B_y)
                if (m_ab < m_ap and m_ap < m_ad):
                    goal_flag = False                                
            elif (section == 2):
                # Slope of line AD
                m_ad = self.slope(D_x, D_y, A_x, A_y)
                # Slope of point A and point P
                m_ap = self.slope(A_x, A_y, x_g, y_g)
                # Slope of point B and point P
                m_bp = self.slope(B_x, B_y, x_g, y_g)
                # Slope of point B and point C
                m_bc = self.slope(B_x, B_y, C_x, C_y)
                if (m_ad > m_ap and m_bp > m_bc):
                    goal_flag = False
            elif (section == 3):
                # Slope of line AD
                m_ad = self.slope(D_x, D_y, A_x, A_y)
                # Slope of point A and point P
                m_ap = self.slope(A_x, A_y, x_g, y_g)
                # Slope of point D and point P
                m_dp = self.slope(D_x, D_y, x_g, y_g)
                # Slope of point B and point C
                m_bc = self.slope(B_x, B_y, C_x, C_y)
                # Slope of point C and point D
                m_cd = self.slope(D_x, D_y, C_x, C_y)
                if (m_cd > m_dp and m_dp > m_ad):
                    goal_flag = False
            #print("goal :", x_g, y_g )
            #print('-------------------------------------')
            if (goal_flag == False):
                #print("section :", section)
                return x_g[0], y_g[0]

    # Additional function
    def get_obj_slide_right(self,tl, dl):
        x_square = (dl + self.w0 / 2.) * np.cos(tl) + (self.w0 / 2. + self.fw) * np.sin(tl) # x_sq (Center of the object)
        y_square = (dl + self.w0 / 2.) * np.sin(tl) - (self.w0 / 2. + self.fw) * np.cos(tl) # y_sq (Center of the object)
        x_square = x_square*2.5 + self.center_coord[0]*2 
        y_square = y_square*2.5
        return x_square, y_square    # *2.5 for scaling up output
    
    # Additional function
    def get_obj_slide_left(self,tr, dr):
        x_square = self.wp + (dr + self.w0 / 2.) * np.cos(np.float64(tr)) - (self.fw + self.w0 / 2.) * np.sin(np.float64(tr)) 
        y_square = (dr + self.w0 / 2.) * np.sin(np.float64(tr)) + (self.fw + self.w0 / 2.) * np.cos(np.float64(tr))
        obj_center = np.array([x_square, y_square])
        obj_center = obj_center*2.5
        obj_center += self.center_coord
        return obj_center    # *2.5 for scaling up output

def callback(_locals, _globals, num_episodes = 30000):
    """
    Callback is called at each step (timestep) for DQN
    :param_locals: (dict)
    :param_globals: (dict)
    """
    global n_steps, best_mean_reward
        
    if (env.episode) > num_episodes:
        return False
    
    # Returning false will stop training early
    return True

if __name__ == '__main__':

    # Instantiate the env
    env_test = FFEnv()
    env = FFEnv()

    # Check the env
    # check_env(env_test, warn=True)

    # Wrap it
    env = Monitor(env, filename=None, allow_early_resets=True)

    # Define the model
    # Custom MLP Policy 
    policy_kwargs = dict(net_arch=[256, 256, 256])

    model = PPO2('MlpPolicy', env, n_steps = 256, policy_kwargs=policy_kwargs, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(1e100), callback=callback)

    model.save("ppo2_her_train_6")

    # Plotting the scores

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(env.scores)), env.scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('score_vs_eps.png')
    # plt.show()

    # Plotting 
    Y = np.asarray(env.log.get_log('scores'))
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('episode return')
    plt.savefig('eps_return_vs_eps.png')
    # plt.show()