"""
Author: Yang Hanlin
Date:   2021/10/22
Description: instantiate object
"""
class EntityState:
    """
    Physical/external base state of all entities.
    """
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class Entity:
    """
    Properties and state of physical world entity.
    """
    def __init__(self):
        self.name = ''
        self.movable = False
        self.collide = False
        self.color = None
        self.max_speed = None
        self.state = EntityState()
        self.initial_mass = 1.0


class Landmark(Entity):
    """
    Properties of landmark entities.
    """
    def __init__(self):
        super(Landmark, self).__init__()
