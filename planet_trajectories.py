# -*- coding: utf-8 -*-
"""
Calculate planet trajectories for one planet and sun.
"""

## imports --------------------------------------------------------------------
import numpy as np



## global constants -----------------------------------------------------------
G = 6.67408e-11



## Body class defenition ------------------------------------------------------
class Body():
    """
    Defines a body. Also provides the compute_acceleration function to compute
    accelerations induced by this body.
    """
    
    def __init__(self, name, pos_x, pos_y, v_x, v_y, mass, radius):
        """
        Initialises a body.
        
        INPUT:
            name (string): name of the body
            pos_x (number): x-coordinate of body location [m]
            pos_y (number): y-coordinate of body location [m]
            v_x (number): x component of body velocity [m/s]
            v_y (number): y component of body velocity [m/s]
            mass (number): mass of body [kg]
            radius (number): radius of body [m]
        """
        self.name = name
        self.m    = mass
        self.r    = radius
        
        self.x    = pos_x
        self.y    = pos_y
        
        self.v_x  = v_x
        self.v_x  = v_x
    
    
    def __str__(self):
        """
        Overwrite __str__ to print the bodies name
        """
        return "Body class object named: " + self.name
    
    
    def compute_acceleration(self, x, y):
        """
        Returns the acceleration induced by the body on an object at location
        (x,y).
        
        INPUT:
            x (number/array): x-coordinate of object location
            y (number/array): y-coordinate of object location
        
        OUTPUT (tuple(a_x, a_y)):
            a_x (number/array): x component of acceleration
            a_y (number/array): y component of acceleration
        """
        a_x = G * self.m / (x*x + y*y) * -x/(np.sqrt(x*x + y*y))
        a_y = G * self.m / (x*x + y*y) * -y/(np.sqrt(x*x + y*y))
        return a_x, a_y



## defenition of bodies -------------------------------------------------------
sun_mass = 1.989e30
sun_radius = 695700000.0
sun = Body("Sun", 0, 0, 0, 0, sun_mass, sun_radius)

ax, ay = sun.compute_acceleration(1000000000.0, 500000000.0)

print("ax:",ax)
print("ay:",ay)












