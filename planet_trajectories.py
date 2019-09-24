# -*- coding: utf-8 -*-
"""
Calculate planet trajectories for one planet and sun.
"""

## imports --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt



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
        
        OUTPUT (numpy_array([a_x, a_y])):
            a_x (number/array): x component of acceleration [m/s^2]
            a_y (number/array): y component of acceleration [m/s^2]
        """
        a_x = G * self.m / (x*x + y*y) * -x/(np.sqrt(x*x + y*y))
        a_y = G * self.m / (x*x + y*y) * -y/(np.sqrt(x*x + y*y))
        return np.array([a_x, a_y])
    
    
    def step(self, dt, bodies):
        """
        Calculates new position and velocity after time dt. Uses numerical
        integration with a 4th-order Runge-Kutta integrator.
        
        INPUT:
            dt (number): timestep of integration [s]
            bodies (list of Body object): bodies influencing the current body
        
        OUTPUT:
            x_n1 (number): x-coordinate of object location at time t + dt [m]
            y_n1 (number): y-coordinate of object location at time t + dt [m]
            vx_n1 (number): x component of velocity at time t + dt [m/s]
            vy_n1 (number): y component of velocity at time t + dt [m/s]
        """
        k1x = self.v_x
        k1y = self.v_y
        
        k1vx, k1vy = sum([body.compute_acceleration(self.x, self.y) for body \
                          in bodies])
        
        k2x = self.v_x + dt/2 * k1vx
        k2y = self.v_y + dt/2 * k1vy
        
        k2vx, k2vy = sum([body.compute_acceleration(self.x + dt/2 * k1x, 
                                                    self.y + dt/2 * k1y) for \
                          body in bodies])
        
        k3x = self.v_x + dt/2 * k2vx
        k3y = self.v_y + dt/2 * k2vy
        
        k3vx, k3vy = sum([body.compute_acceleration(self.x + dt/2 * k2x, 
                                                    self.y + dt/2 * k2y) for \
                          body in bodies])
        
        k4x = self.v_x + dt * k3vx
        k4y = self.v_y + dt * k3vy
        
        k4vx, k4vy = sum([body.compute_acceleration(self.x + dt * k3x, 
                                                    self.y + dt * k3y) for \
                          body in bodies])
        
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

# use floats for the x positions to avoid integer overflow in C, causing an 
# error in numpy which result in a weird attribute error ('int' object has no 
# attribute 'sqrt')
sun     = Body("Sun", 0, 0, 0, 0, sun_mass, sun_radius)
mercury = Body("Mercury", -46000000000., 0, 0, -58980, 0.33011e24, 2439700)
venus   = Body("Venus", -107480000000., 0, 0, -35260, 4.8675e24, 6051800)
earth   = Body("Earth", earth_x0, earth_y0, earth_vx0, earth_vy0, earth_mass,
               earth_radius)
mars    = Body("Mars", -206620000000., 0, 0, -26500, 6.4171e23, 3389500)
jupiter = Body("Jupiter", -740520000000., 0, 0, -13720, 1898.19e24, 71492000)
saturn  = Body("Saturn", -1352550000000., 0, 0, -10180, 568.34e24, 54364000)
uranus  = Body("Uranus", -2741300000000., 0, 0, -7110, 86.813e24, 24973000)
neptune = Body("Neptune", -4444450000000., 0, 0, -5500, 102.413e24, 24341000)

bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]



## simulation and output parameters -------------------------------------------
dt = 86400.0 # Earth day in seconds

N_steps = 365

filename = "trajectories.txt"

plot = True # create a plot or not? simulation will take some extra time



## Main run -------------------------------------------------------------------
# setup file front matter
sim_setup_str = "NUM_BODIES \n{}\n\nNUM_STEPS\n{}\n\n".format(len(bodies), 
                                                              N_steps)

names, masses, radii = [], [], []
for planet in bodies:
    names.append(planet.name)
    masses.append(str(planet.m))
    radii.append(str(planet.r))
sim_input_str = "NAMES\n{}\n\nMASSES\n{}\n \
\nRADII\n{}\n\nTRAJECTORIES\n".format("\n".join(names), "\n".join(masses), 
                                      "\n".join(radii))

front_matter = sim_setup_str + sim_input_str

# initial state
xys = []
for body in bodies:
    xys.append(str(body.x))
    xys.append(str(body.y))
trajectories_str = "0, " + ", ".join(xys) + "\n"

# prepare a plot
if plot:
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Trajectories")
    ax.set(xlabel="x-coordinate [m]", ylabel="y-coordinate [m]")
    
    colours = ["yellow", "black", "green", "blue", "red", "grey", "orange", 
               "pink", "lightblue"]
    
    xy_plot = np.zeros((N_steps, len(bodies), 2))

# run trajectories
for i in range(N_steps):
    
    for k in range(1, len(bodies)):
        # calculate new x and y coordinates for each body
        bodies[k].step(dt, bodies[:k]+bodies[k+1:])
    
    xys = []
    for j in range(len(bodies)):
        # data for output file
        xys.append(str(bodies[j].x))
        xys.append(str(bodies[j].y))
        
        # add data to plot arrays if desired
        if plot:
            xy_plot[i, j, :] = bodies[j].x, bodies[j].y
    
    trajectories_str += str(i+1) + ", " + ", ".join(xys) + "\n"

# write output
with open(filename, "w+") as f:
    f.write(front_matter)
    f.write(trajectories_str)
f.close()

if plot:
    for j in range(len(bodies)):
        ax.plot(xy_plot[:,j,0], xy_plot[:,j,1], linestyle="none", marker="x", 
                color=colours[j], label=bodies[j].name)
    ax.legend(loc="upper left")
    fig.show()





