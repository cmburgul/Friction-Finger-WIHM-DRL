import numpy as np
import pyglet
import random
import time
from pyglet.window import key
from math import radians, degrees

class FFEnv(object):
    viewer = None
    dt = 0.05    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100.}
    state_dim = 2
    action_dim = 2

    w0 = 25  # Object width
    wp = 50  #  
    fw = 18  # Finger width

    def __init__(self):
        self.ff_info = np.zeros(2, dtype=[('d', np.float32), ('t', np.float32), ('a', np.int)])
        self.ff_info['t'][0] = radians(60)
        self.ff_info['d'][0] = 90 # Min 35mm - Max 120mm
        self.ff_info['a'] = 0

    def step(self, action):
        done = False
        r = 0.

        # Action : Sliding on Right Finger
        # Give Left Finger Pararmeters and get Right Finger Paramters -> Call slide Right finger 
        action = np.clip(action, *self.action_bound) # Clipping the action
        
        # Creating dummy variables t0, d0, t1, d1
        self.t0 = self.ff_info['t'][0]
        self.d0 = self.ff_info['d'][0]

        self.t0 += action * self.dt # Adding the action delta theta to theta_right or t_r

        print('t0 : ', degrees(self.t0))
        print('d0 : ', self.d0)

        # Constraining the left finger to lesser than 140 deg
        if ( self.t0 > radians(140) ):
            action = 0
        if (self.d0 >= float(105)):
            action = 0

        # Getting tr, dr by giving tl, d1
        self.t1, self.d1 = self.calc_right_config(self.t0, self.d0)
        print('dr : -----------', self.d1)
        # Constraining t1, d1 limits
        if ( self.d1 > float(95) ):
            action = 0
        if ( self.t1 <= radians(40) or self.t1 >= radians(152) ): # greater than 140 deg
            action = 0

        if (action != 0):
            self.ff_info['t'][0] += action * self.dt  
            #self.ff_info['t'][0] %= np.pi * 2 # normalize ! Why ? Need to know         
            # Getting tl, dl by giving tr, dr
            self.ff_info['t'][1], self.ff_info['d'][1] = self.calc_right_config(self.ff_info['t'][0], self.ff_info['d'][0]) 

        # For only sliding on left finger code we no need use ff_info['a']
        # state
        s = self.ff_info['t'] # As of now only consider theta in states
        # done and reward
        done = False
        r = 0. # Sparse reward
        return s, r, done
    
    def reset(self):
        self.ff_info['t'] = 2 * np.pi/2
        return self.ff_info['t']

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.ff_info, self.goal)
        self.viewer.render()

    def calc_right_config(self, tl, dl):
        
        #print("tl : ", degrees(tl))
        #print("dl : ", dl)
        d1v = np.array([dl * np.cos(tl), dl * np.sin(tl)])
        d1v = d1v.reshape(1,2)[0]
        #print("d1v : ", d1v)
        w0v = np.array([self.w0 * np.sin(tl), -self.w0 * np.cos(tl)])
        w0v = w0v.reshape(1,2)[0]
        print("w0v : ", w0v)
        wpv = np.array([self.wp, 0.])
        #print("wpv : ", wpv)
        f1v = np.array([self.fw * np.sin(tl), -self.fw * np.cos(tl)])
        f1v = f1v.reshape(1,2)[0]
        #print("f1v : ", f1v)
        av = d1v + w0v + f1v - wpv
        #print("av :", av)
        dr = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        #print("dr :", dr)
        tr = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(self.fw, dr)
        return tr, dr

    def sample_action(self):
        return np.random.rand(1)-0.5    # One action

class Viewer(pyglet.window.Window):
    w0 = 25  # Object width
    wp = 50  # 
    fw = 18  # Finger width

    def __init__(self, ff_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=500, height=500, resizable=False, caption='FrictoinFinger', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.ff_info = ff_info

        self.center_coord = np.array([200, 0]) 
        # Calling translateLeft_obj and translateLeft_fingers just to initialize the position of object and finger positions
        # Seperate function
        self.obj_pos, obj_center = self.slide_Right_obj(self.ff_info['t'][0], self.ff_info['d'][0]) + self.center_coord
        self.finger_l, self.finger_r = self.slide_Right_fingers(self.ff_info['t'][0], self.ff_info['d'][0]) + self.center_coord
        
        self.batch = pyglet.graphics.Batch()  # display whole batch at once

        self.object = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [self.obj_pos[1][0], self.obj_pos[1][1],         # location
                     self.obj_pos[0][0], self.obj_pos[0][1],
                     self.obj_pos[3][0], self.obj_pos[3][1],
                     self.obj_pos[2][0], self.obj_pos[2][1]]),
            ('c3B', (86, 109, 249) * 4))    # color
        
        self.finger_l = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.finger_l[3][0], self.finger_l[3][1],
                     self.finger_l[2][0], self.finger_l[2][1],
                     self.finger_l[1][0], self.finger_l[1][1],
                     self.finger_l[0][0], self.finger_l[0][1]]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.finger_r = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.finger_r[3][0], self.finger_r[3][1],
                     self.finger_r[2][0], self.finger_r[2][1],
                     self.finger_r[1][0], self.finger_r[1][1],
                     self.finger_r[0][0], self.finger_r[0][1] ]),
                     ('c3B', (249, 86, 86) * 4,))

    def slide_Right_obj(self,tl, dl):
        # Input : 
        # tl : theta_left
        # dl : left distance from object to origin
        # Output : Left and Right Finger Position 
        
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
        obj_center = np.vstack([x_square, y_square])
        return pts*2.5, obj_center*2.5

    def slide_Right_fingers(self, tl, dl):

        # Calculate theta1, dl
        d1v = np.array([dl * np.cos(tl), dl * np.sin(tl)])
        w0v = np.array([self.w0 * np.sin(tl), -self.w0 * np.cos(tl)])
        print("w0v : ", w0v)
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * np.sin(tl), -self.fw * np.cos(tl)])
        av = d1v + w0v + f1v - wpv
        
        # Calculated Values of thetar, dr
        dr = np.sqrt(float((av * av).sum() - self.fw * self.fw))
        tr = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(self.fw, dr)
        #print("tr : ", degrees(tr))
        #print("dr : ", dr)

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
        #print("fw_1 :", fw_1)
        #print("fw_2 :", fw_2)
        return fw_1*2.5, fw_2*2.5

    def render(self):
        self._update_finger()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
 
    def _update_finger(self):
        # Action is Sliding on Right finger
        #print('Actuating Right Finger')
        obj_pos_, obj_center = self.slide_Right_obj(self.ff_info['t'][0], self.ff_info['d'][0])
        finger_l_, finger_r_ = self.slide_Right_fingers(self.ff_info['t'][0], self.ff_info['d'][0])

        obj_pos_ += self.center_coord
        finger_l_ += self.center_coord
        finger_r_ += self.center_coord
        
        self.object.vertices = np.hstack([obj_pos_[1][0], obj_pos_[1][1],         
                                      obj_pos_[0][0], obj_pos_[0][1],
                                      obj_pos_[3][0], obj_pos_[3][1],
                                      obj_pos_[2][0], obj_pos_[2][1]])
        self.finger_l.vertices = np.hstack([finger_l_[3][0], finger_l_[3][1],
                                                 finger_l_[2][0], finger_l_[2][1],
                                                 finger_l_[1][0], finger_l_[1][1],
                                                 finger_l_[0][0], finger_l_[0][1]])
        self.finger_r.vertices = np.hstack([finger_r_[3][0], finger_r_[3][1],
                                                 finger_r_[2][0], finger_r_[2][1],
                                                 finger_r_[1][0], finger_r_[1][1],
                                                 finger_r_[0][0], finger_r_[0][1]])

if __name__ == '__main__':
    env = FFEnv()
    count = 0
    #start_time = time.time()
    while True:
        env.render()
        #env.step(env.sample_action())

        """
        if (count >= 3):
            break
        count += 1
        """

        """
        count += 1
        if (time.time() - start_time >= 1):
            print("count : ", count)
            break
        """