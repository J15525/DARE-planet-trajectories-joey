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
        self.v_y  = v_y
    
    
    def __str__(self):
        """
        Overwrite __str__ to print the bodies name and the class name
        """
        return "Body class object named: " + self.name
    
    
    def compute_acceleration(self, x, y):
        """
        Returns the acceleration induced by the body on an object at location
        (x,y).
        
        INPUT:
            x (number/array): x-coordinate of object location [m]
            y (number/array): y-coordinate of object location [m]
        
        OUTPUT (tuple(a_x, a_y)):
            a_x (number/array): x component of acceleration [m/s^2]
            a_y (number/array): y component of acceleration [m/s^2]
        """
        a_x = G * self.m / (x*x + y*y) * -x/(np.sqrt(x*x + y*y))
        a_y = G * self.m / (x*x + y*y) * -y/(np.sqrt(x*x + y*y))
        return a_x, a_y
    
    
    def step(self, dt, body):
        """
        Calculates new position and velocity after time dt. Uses numerical
        integration with a 4th-order Runge-Kutta integrator.
        
        INPUT:
            dt (number): timestep of integration [s]
        
        OUTPUT:
            x_n1 (number): x-coordinate of object location at time t + dt [m]
            y_n1 (number): y-coordinate of object location at time t + dt [m]
            vx_n1 (number): x component of velocity at time t + dt [m/s]
            vy_n1 (number): y component of velocity at time t + dt [m/s]
        """
        k1x = self.v_x
        k1y = self.v_y
        
        k1vx, k1vy = body.compute_acceleration(self.x, self.y)
        
        k2x = self.v_x + dt/2 * k1vx
        k2y = self.v_y + dt/2 * k1vy
        
        k2vx, k2vy = body.compute_acceleration(self.x + dt/2 * k1x, 
                                               self.y + dt/2 * k1y)
        
        k3x = self.v_x + dt/2 * k2vx
        k3y = self.v_y + dt/2 * k2vy
        
        k3vx, k3vy = body.compute_acceleration(self.x + dt/2 * k2x, 
                                               self.y + dt/2 * k2y)
        
        k4x = self.v_x + dt * k3vx
        k4y = self.v_y + dt * k3vy
        
        k4vx, k4vy = body.compute_acceleration(self.x + dt * k3x, 
                                               self.y + dt * k3y)
        
        x_n1 = self.x + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        y_n1 = self.y + dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
        vx_n1 = self.v_x + dt/6 * (k1vx + 2*k2vx + 2*k3vx + k4vx)
        vy_n1 = self.v_y + dt/6 * (k1vy + 2*k2vy + 2*k3vy + k4vy)
        
        self.x  = x_n1
        self.y  = y_n1
        self.v_x = vx_n1
        self.v_y = vy_n1



## defenition of bodies -------------------------------------------------------
sun_mass   = 1.989e30
sun_radius = 695700000.0

earth_x0     = -147095000000.0
earth_y0     = 0.0
earth_vx0    = 0.0
earth_vy0    = -30300.0
earth_mass   = 5.972e24
earth_radius = 6371000.0

sun   = Body("Sun", 0, 0, 0, 0, sun_mass, sun_radius)
earth = Body("Earth", earth_x0, earth_y0, earth_vx0, earth_vy0, earth_mass,
             earth_radius)

bodies = [sun, earth]



## simulation parameters ------------------------------------------------------
dt = 86400.0 # Earth day in seconds

N_steps = 365



## test cases -----------------------------------------------------------------
# =============================================================================
# # case 1
# ax, ay = sun.compute_acceleration(1000000000.0, 500000000.0)
# 
# print("ax:",ax)
# print("ay:",ay)
# 
# 
# # case 2
# earth.step(dt, sun)
# 
# print("x:",earth.x)
# print("y:",earth.y)
# print("vx:",earth.vx)
# print("vy:",earth.vy)
# =============================================================================


# case 3
sim_setup_str = "NUM_BODIES \n{}\n\nNUM_STEPS\n{}\n\n".format(len(bodies), N_steps)
names, masses, radii = [], [], []
for planet in bodies:
    names.append(planet.name)
    masses.append(str(planet.m))
    radii.append(str(planet.r))
sim_input_str = "NAMES\n{}\n\nMASSES\n{}\n\nRADII\n{}\n\nTRAJECTORIES\n".format("\n".join(names), "\n".join(masses), "\n".join(radii))
front_matter = sim_setup_str + sim_input_str

trajectories_str = "0, {}, {}, {}, {}\n".format(sun.x, sun.y, earth.x, earth.y)
# run trajectories
for i in range(N_steps):
    earth.step(dt, sun)
    trajectories_str += "{}, {}, {}, {}, {}\n".format(i+1, sun.x, sun.y, earth.x, earth.y)

with open("trajectories.txt", "w+") as f:
    f.write(front_matter)
    f.write(trajectories_str)
f.close()

    






