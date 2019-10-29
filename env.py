import pyglet
import numpy as np

class FrictionFinger(object):
    viewer = None
    d2 = 5.0
    t2 = np.pi / 2
    t1 = np.pi / 2
    d1 = 6.2

    def __init__(self):
        pass

    def step(self,action):
        pass

    def reset(self):
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer()
        self.viewer.render()

class Viewer(pyglet.window.Window):
    link_thickness = None

    def __init__(self):
        super(Viewer, self).__init__(width=400,
                                     height=400,
                                     resizable=False,
                                     caption='Friction Finger',
                                     vsync=False)

        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.batch = pyglet.graphics.Batch() # To display whole batch at once
        self.object = self.batch.add(4,       # Adding 4 points in a batch
                                    pyglet.gl.GL_QUADS,
                                    None,
                                    ('v2f', [150,125, 275,125, 275,200, 200,200]),
                                    ('c3B', (86, 109, 249)*4))
        self.finger_l = self.batch.add(4,
                                   pyglet.gl.GL_QUADS, 
                                   None,
                                   ('v2f', [150,25, 200,25, 200,250, 150,250]),
                                   ('c3B', (249, 86, 86)*4))
        self.finger_r = self.batch.add(4,
                                   pyglet.gl.GL_QUADS,
                                   None,
                                   ('v2f', [275,25, 325,25, 325,250, 275,250]),
                                   ('c3B', (249,86,86)*4))

    def render(self):
        self._update_config()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_config(self):
        pass

if __name__ == '__main__':
    env = FrictionFinger()
    while True:
        env.render()
        


