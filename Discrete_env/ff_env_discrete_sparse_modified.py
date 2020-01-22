# Sparse reward and Modified State space

import numpy as np
import pyglet
import random
import time
import matplotlib.pyplot as plt
from math import radians, degrees

MAX_OBJ_LIMIT = 105 # in mm
MIN_OBJ_LIMIT = 25  # in mm
Δθ = 1 # Actuation magnitude
 
# tl : theta_left
# dl : distance from object base to left finger base
# tr : theta_right
# dr : distance from object base to right finger base

class FFEnv(object):
    viewer = None
    dt = 0.01    # refresh rate
    #action_bound = [-1, 1]
    state_size = 7
    action_size = 5
    center_coord = np.array([100, 0])

    w0 = 25  # Object width
    wp = 50  # 
    fw = 18  # Finger width

    def __init__(self):
        self.ff_info = np.zeros(2, dtype=[('d', np.float32), ('t', np.float32), ('a', np.int)])
        self.goal = {'x': 0., 'y': 0., 'w':self.w0} # Goal Position of the Object 
        #self.goal['x'], self.goal['y'] = self.get_goal_point()
        self.goal['x'], self.goal['y'] = 388, 228
        
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
        print('Initial ff_info : ', self.ff_info)
        
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
        
        # done and reward
        #print("dist_x :", dist_x)
        #print("dist_y :", dist_y)
        
        # reward engineering
        #reward = -np.sqrt((dist_x**2 + dist_y**2))/200
        #print(reward)
        # Sparse reward
        reward = -1 # Sparse reward Calculate distance between obj_pos and goal_pos

        # Check if object is near to goal in x-axis
        if ( self.goal['x'] - self.goal['w']/2 < self.obj_pos['x'] < self.goal['x'] + self.goal['w']/2 ):
            # Check if object is near to goal in y-axis
            if ( self.goal['y'] - self.goal['w']/2 < self.obj_pos['y'] < self.goal['y'] + self.goal['w']/2 ):
                reward += 1.
                self.on_goal += 1
                if self.on_goal > 1:
                    done = True
        else:
            self.on_goal = 0
    
        # Concatenate and normalize
        """
        next_state = np.concatenate((self.ff_info['t'][0], 
                                     self.ff_info['t'][1], 
                                     self.obj_pos['x']/200, 
                                     self.obj_pos['y']/200, 
                                     dist_x/200, 
                                     dist_y/200, 
                                     [1. if self.on_goal else 0.]), axis=None)
        """
        next_state = np.concatenate((self.ff_info['t'][0], 
                                     self.ff_info['t'][1], 
                                     self.obj_pos['x']/200, 
                                     self.obj_pos['y']/200, 
                                     self.goal['x']/200, 
                                     self.goal['y']/200, 
                                     [1. if self.on_goal else 0.]), axis=None)

        #next_state = np.concatenate((self.ff_info['t'][0], self.ff_info['t'][1], self.obj_pos['x']/200, self.obj_pos['y']/200, self.goal['x']/200, self.goal['y']/200, [1. if self.on_goal else 0.]), axis=None)
        #print("state -> step(action): ", next_state)
        #print('ff_info : ', self.ff_info)
        
        return next_state, reward, done
    
    def reset(self):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # During start of every episode the agent will reset the envirohnment 
        # 1. Gives a new goal location
        # 2. Gives a initial state of the system 
        #
        # Input : none
        # Output : state
        # State : { theta_l, theta_r, O_x, O_y, (G-O)_x, (G-O)_y, done }
        self.ff_info['t'][0] = radians(90)
        self.ff_info['d'][0] = 35
        self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0])
        
        # Goal location 
        # self.goal['x'], self.goal['y'] = self.get_goal_point() # Multi-Goal RL
        self.goal['x'], self.goal['y'] = 388, 228
        
        # Object Position
        # There is a difference in getting object position with slide_obj_right and slide_obj_left functions 
        # Let's follow a standard of slide_obj_right
        self.obj_pos['x' ], self.obj_pos['y'] = self.get_obj_slide_right(self.ff_info['t'][0], self.ff_info['d'][0])        
        #obj_pos_slide_left = self.get_obj_slide_left(self.ff_info['t'][1], self.ff_info['d'][1])

        #print("Obj Center while sliding right : ", self.obj_pos)
        #print("Obj Center while sliding left : ", obj_pos_slide_left)

        # Distance from goal to object
        dist_x = self.obj_pos['x'] - self.goal['x'] 
        dist_y = self.obj_pos['y'] - self.goal['y']

        # Goal Positions
        #self.goal['x']
        #self.goal['y']

        #state = np.concatenate((self.ff_info['t'][0], self.ff_info['t'][1], self.obj_pos['x']/200, self.obj_pos['y']/200, dist_x/200, dist_y/200, [1. if self.on_goal else 0.]), axis=None)

        state = np.concatenate((self.ff_info['t'][0], 
                                self.ff_info['t'][1], 
                                self.obj_pos['x']/200, 
                                self.obj_pos['y']/200, 
                                self.goal['x']/200, 
                                self.goal['y']/200, 
                                [1. if self.on_goal else 0.]), axis=None)        
        
        #print("state : ", state)
        return state

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.ff_info, self.goal)
        self.viewer.render()
        #print('===============x================x====================x====================')

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

class Viewer(pyglet.window.Window):
    w0 = 25  # Object width
    wp = 50  # 
    fw = 18  # Finger width

    def __init__(self, ff_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='FrictoinFinger', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.ff_info = ff_info
        self.center_coord = np.array([100, 0]) 

        self.obj_pos, self.obj_center = self.slide_Left_obj(self.ff_info['t'][1], self.ff_info['d'][1]) 
        self.finger_l, self.finger_r = self.slide_Left_fingers(self.ff_info['t'][1], self.ff_info['d'][1]) 

        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.goal = goal
        #self.goal['x'] += 2*self.center_coord[0]
        
        self.obj_center_pos = {'x':0., 'y':0.} # Object Position
        self.obj_loc_x = np.array([]) # object position x
        self.obj_loc_y = np.array([]) # object position y

        # Object Position
        self.object = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [self.obj_pos[1][0], self.obj_pos[1][1],     
                     self.obj_pos[0][0], self.obj_pos[0][1],
                     self.obj_pos[3][0], self.obj_pos[3][1],
                     self.obj_pos[2][0], self.obj_pos[2][1]]),
            ('c3B', (138, 43, 226) * 4))    # color
        # Left Finger
        self.finger_l = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.finger_l[3][0], self.finger_l[3][1],
                     self.finger_l[2][0], self.finger_l[2][1],
                     self.finger_l[1][0], self.finger_l[1][1],
                     self.finger_l[0][0], self.finger_l[0][1]]),
            ('c3B', (255, 215, 0) * 4,))    # color
        # Right finger
        self.finger_r = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.finger_r[3][0], self.finger_r[3][1],
                     self.finger_r[2][0], self.finger_r[2][1],
                     self.finger_r[1][0], self.finger_r[1][1],
                     self.finger_r[0][0], self.finger_r[0][1] ]),
                     ('c3B', (255, 215, 0) * 4,))
        # Goal Position of the object
        self.goal_pos = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [goal['x']+20, goal['y']+20,
                     goal['x']+20, goal['y']-20,
                     goal['x']-20, goal['y']-20,
                     goal['x']-20, goal['y']+20 ]),
                     ('c3B', (124, 252, 0) * 4,))
        #print("obj_pos : ", self.obj_pos)
        #print("obj_goal : ", self.obj_goal)             
        print("goal pos : ", goal['x'], goal['y'])
        #obj_loc = np.array([self.obj_center])

    def slide_Left_obj(self,tr, dr):
        # Define :  slide the object on left finger with friction enabled on right finger and disabled on left finger
        # Input : t_right, d_right
        # Output : t_left, d_left

        # Transformation Matrix
        x_square = self.wp + (dr + self.w0 / 2.) * np.cos(np.float64(tr)) - (self.fw + self.w0 / 2.) * np.sin(np.float64(tr)) 
        y_square = (dr + self.w0 / 2.) * np.sin(np.float64(tr)) + (self.fw + self.w0 / 2.) * np.cos(np.float64(tr))
        pts = np.array([[-self.w0 / 2., -self.w0 / 2., self.w0 / 2., self.w0 / 2.], [-self.w0 / 2., self.w0 / 2., self.w0 / 2., -self.w0 / 2.], [1, 1, 1, 1]])
        R = np.array([[np.cos(tr), -np.sin(tr), x_square], [np.sin(tr), np.cos(tr), y_square], [0, 0, 1]])
        
        # Points after transformation
        pts_new = np.dot(R, pts)
        
        # Plotting the Object
        pts = np.transpose([[pts_new[0, :]], [pts_new[1, :]]])
        pts = pts.reshape((4, 2))
        obj_center = np.array([x_square, y_square])
        return pts*2.5, obj_center*2.5 # *2.5 for scaling up output

    def slide_Left_fingers(self,tr, dr):
        
        # Calculate thetar, dr
        d2v = np.array([dr * np.cos(np.float64(tr)), dr * np.sin(np.float64(tr))])
        w0v = np.array([self.w0 * np.sin(np.float64(tr)), -self.w0 * np.cos(np.float64(tr))])
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * np.sin(np.float64(tr)), -self.fw * np.cos(np.float64(tr))])
        av = d2v - f1v - w0v + wpv
        
        # Calculated Values of theta1, dl
        dl = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        tl = np.arctan2(float(av[1]), float(av[0])) + np.arctan2(self.fw, dl)
        l_fw_pts = np.array([[0., 0., self.fw, self.fw], [10, 130, 130, 10], [1.0, 1.0, 1.0, 1.0]])
        r_fw_pts = np.array([[0., 0., -self.fw, -self.fw], [10, 130, 130, 10], [1.0, 1.0, 1.0, 1.0]])
        
        # Transformation matrices for the finger width
        R_fw1 = [[np.cos(tl - np.pi / 2.0), -np.sin(tl - np.pi / 2), 0.0], [np.sin(tl - np.pi / 2), np.cos(tl - np.pi / 2), 0.0], [0.0, 0.0, 1.0]]
        R_fw2 = [[np.cos(tr - np.pi / 2), -np.sin(tr - np.pi / 2), self.wp], [np.sin(tr - np.pi / 2), np.cos(tr - np.pi / 2), 0.0], [0.0, 0.0, 1.0]]
        
        # finger Coordinates 1-> Left, 2-> Right
        pts_fw1 = np.dot(R_fw1, l_fw_pts)
        pts_fw2 = np.dot(R_fw2, r_fw_pts)
        
        # Plotting the fingers
        fw_1 = np.transpose([[pts_fw1[0, :]], [pts_fw1[1, :]]]).reshape((4, 2))
        fw_2 = np.transpose([[pts_fw2[0, :]], [pts_fw2[1, :]]]).reshape((4, 2))
        
        return fw_1*2.5, fw_2*2.5 # *2.5 for scaling up output

    def slide_Right_obj(self,tl, dl):
        # Define :  slide the object on right finger with friction enabled on left finger and disabled on right finger
        # Input : t_left, d_left
        # Output : t_right, d_right
        
        # Transformation Matrix
        x_square = (dl + self.w0 / 2.) * np.cos(tl) + (self.w0 / 2. + self.fw) * np.sin(tl) # x_sq (Center of the object)
        y_square = (dl + self.w0 / 2.) * np.sin(tl) - (self.w0 / 2. + self.fw) * np.cos(tl) # y_sq (Center of the object)
        pts = np.array([[-self.w0 / 2., -self.w0 / 2., self.w0 / 2., self.w0 / 2.], [-self.w0 / 2., self.w0 / 2., self.w0 / 2., -self.w0 / 2.], [1, 1, 1, 1]])
        R = np.array([[np.cos(tl), -np.sin(tl), x_square], [np.sin(tl), np.cos(tl), y_square], [0, 0, 1]])
    
        # Points after transformation
        pts_new = np.dot(R, pts)
        
        # Plotting the Object
        pts = np.transpose([[pts_new[0, :]], [pts_new[1, :]]])
        pts = pts.reshape((4, 2))
        obj_center = np.array([x_square, y_square])
    
        return pts*2.5, obj_center*2.5 # *2.5 for scaling up output

    def slide_Right_fingers(self, tl, dl):
        
        # Calculate theta1, dl
        d1v = np.array([dl * np.cos(tl), dl * np.sin(tl)])
        w0v = np.array([self.w0 * np.sin(tl), -self.w0 * np.cos(tl)])
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * np.sin(tl), -self.fw * np.cos(tl)])
        av = d1v + w0v + f1v - wpv
        
        # Calculated Values of thetar, dr
        dr = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        tr = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(self.fw, dr)

        l_fw_pts = np.array([[0., 0., self.fw, self.fw], [10, 130, 130, 10], [1.0, 1.0, 1.0, 1.0]])
        r_fw_pts = np.array([[0., 0., -self.fw, -self.fw], [10, 130, 130, 10], [1.0, 1.0, 1.0, 1.0]])
        # Transformation matrices for the finger width
        R_fw1 = [[np.cos(tl - np.pi / 2.0), -np.sin(tl - np.pi / 2), 0.0], [np.sin(tl - np.pi / 2), np.cos(tl - np.pi / 2), 0.0], [0.0, 0.0, 1.0]]
        R_fw2 = [[np.cos(tr - np.pi / 2), -np.sin(tr - np.pi / 2), self.wp], [np.sin(tr - np.pi / 2), np.cos(tr - np.pi / 2), 0.0], [0.0, 0.0, 1.0]]

        # finger Coordinates 1-> Left, 2-> Right
        pts_fw1 = np.dot(R_fw1, l_fw_pts)
        pts_fw2 = np.dot(R_fw2, r_fw_pts)

        # Plotting the fingers
        fw_1 = np.transpose([[pts_fw1[0, :]], [pts_fw1[1, :]]]).reshape((4, 2))
        fw_2 = np.transpose([[pts_fw2[0, :]], [pts_fw2[1, :]]]).reshape((4, 2))
        
        return fw_1*2.5, fw_2*2.5 # *2.5 for scaling up output

    def render(self):
        self._update_finger()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def get_obj_slide_right(self,tl, dl):
        x_square = (dl + self.w0 / 2.) * np.cos(tl) + (self.w0 / 2. + self.fw) * np.sin(tl) # x_sq (Center of the object)
        y_square = (dl + self.w0 / 2.) * np.sin(tl) - (self.w0 / 2. + self.fw) * np.cos(tl) # y_sq (Center of the object)
        x_square = x_square*2.5 + self.center_coord[0]*2 
        y_square = y_square*2.5
        return x_square, y_square    # *2.5 for scaling up output
 
    def _update_finger(self):
        # Check action in ff_info['a'] and visualize based on it
        obj_pos_, obj_center = self.slide_Right_obj(self.ff_info['t'][0], self.ff_info['d'][0]) 
        finger_l_, finger_r_ = self.slide_Right_fingers(self.ff_info['t'][0], self.ff_info['d'][0])

        self.obj_center_pos['x' ], self.obj_center_pos['y'] = self.get_obj_slide_right(self.ff_info['t'][0], self.ff_info['d'][0])        
        self.obj_loc_x = np.insert(self.obj_loc_x, 0, [self.obj_center_pos['x']])
        self.obj_loc_y = np.insert(self.obj_loc_y, 0, [self.obj_center_pos['y']])
        print('obj_center : ', self.obj_center_pos['x'], self.obj_center_pos['y'], '|', 'goal : ', self.goal)
        #print('goal : ', self.goal)
        print("tl :", degrees(self.ff_info['t'][0]), " | tr :", degrees(self.ff_info['t'][1]), " | dl :", self.ff_info['d'][0], "| dr :", self.ff_info['d'][1],)

        obj_pos_ += self.center_coord
        finger_l_ += self.center_coord
        finger_r_ += self.center_coord

        # Updating obj_pos in graphics
        self.object.vertices = np.hstack([obj_pos_[1][0] + self.center_coord[0], obj_pos_[1][1],         
                                      obj_pos_[0][0] + self.center_coord[0], obj_pos_[0][1],
                                      obj_pos_[3][0] + self.center_coord[0], obj_pos_[3][1],
                                      obj_pos_[2][0] + self.center_coord[0], obj_pos_[2][1]])
        # Updating left finger in graphics
        self.finger_l.vertices = np.hstack([finger_l_[3][0] + self.center_coord[0], finger_l_[3][1],
                                                 finger_l_[2][0] + self.center_coord[0], finger_l_[2][1],
                                                 finger_l_[1][0] + self.center_coord[0], finger_l_[1][1],
                                                 finger_l_[0][0] + self.center_coord[0], finger_l_[0][1]])
        # Updating right finger in graphics
        self.finger_r.vertices = np.hstack([finger_r_[3][0] + self.center_coord[0], finger_r_[3][1],
                                                 finger_r_[2][0] + self.center_coord[0], finger_r_[2][1],
                                                 finger_r_[1][0] + self.center_coord[0], finger_r_[1][1],
                                                 finger_r_[0][0] + self.center_coord[0], finger_r_[0][1]])                                         
    def return_path(self):
        return self.obj_loc_x, self.obj_loc_y


if __name__ == '__main__':
    env = FFEnv()
    count = 0
    while True:
        #print("Action Iteration : ", count)
        env.render()
        #break
        #env.step(env.sample_action())
        #count += 1 
        #print('-------')   
        #env.reset()    
        #print('-------')
        #print("get goal point : ", env.get_goal_point())
        #if (count >= 1):
        #    break
            
        