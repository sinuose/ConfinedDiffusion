import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

class CylinderRandomWalk():
    def __init__ (self, 
                  N,
                  dt,
                  steps,
                  D,
                  R,
                  H):
        
        # Simulation parameters
        self.N      = N         # Number of particles (simultaneous runs)
        self.R      = R         # Cylinder radius
        self.H      = H         # Cylinder height
        self.D      = D         # Diffusion coefficient  [m/s^2]
        self.dt     = dt        # Time step              [s]
        self.steps  = steps     # Number of time steps   

        self.InitializeSimulation()

    def InitializeSimulation(self):
        # Initialize particle positions (r, theta, z)
        self.r = np.sqrt(np.random.uniform(0, 1, self.N)) * self.R
        self.theta = np.random.uniform(0, 2*np.pi, self.N)
        self.z = np.random.uniform(0, self.H, self.N)

        # Convert to Cartesian coordinates
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)

        # current position and trajectory container
        self.currPos = np.array([self.x,self.y,self.z])  # shape: (3, N)
        self.TRAJ = [self.currPos.copy()]  # store each timestep's positions

    def SingleStep(self):
        """
        Perform a single diffusion step for all particles.
        """
        dx = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)
        dy = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)
        dz = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)

        # Update positions
        self.currPos[0] += dx
        self.currPos[1] += dy
        self.currPos[2] += dz

        # Convert to cylindrical for boundary checks
        self.r = np.sqrt(self.currPos[0]**2 + self.currPos[1]**2)
        self.theta = np.arctan2(self.currPos[1], self.currPos[0])

        # Reflective boundary at r = R
        outside = self.r > self.R
        self.r[outside] = 2*self.R - self.r[outside]

        # Reflective boundaries at z = 0 and z = H
        below_zero = self.currPos[2] < 0
        above_H    = self.currPos[2] > self.H

        self.currPos[2][below_zero] = -self.currPos[2][below_zero]
        self.currPos[2][above_H]    = 2*self.H - self.currPos[2][above_H]

        # Convert back to Cartesian
        self.currPos[0] = self.r * np.cos(self.theta)
        self.currPos[1] = self.r * np.sin(self.theta)

        # Save positions at this time step
        self.TRAJ.append(self.currPos.copy())

    def runSimulation(self):
        """
        Run the full simulation for 'steps' time steps.
        """
        for _ in range(self.steps):
            self.SingleStep()
        
    def plotTrajectory(self, fig,ax, particle_index=0):
        """
        Plot the 3D trajectory for a single particle.
        By default, plots the first particle (index 0).
        """
        # Convert list of arrays to one NumPy array of shape (steps+1, 3, N)
        traj_array = np.array(self.TRAJ)

        # Extract x, y, z for the chosen particle
        x_vals = traj_array[:, 0, particle_index]
        y_vals = traj_array[:, 1, particle_index]
        z_vals = traj_array[:, 2, particle_index]

        # Plot the trajectory
        ax.plot(x_vals, y_vals, z_vals, label=f'Particle {particle_index}')


class CircleRandomWalk():
    def __init__ (self, 
                  N,
                  dt,
                  steps,
                  D,
                  R):
        
        # Simulation parameters
        self.N      = N         # Number of particles (simultaneous runs)
        self.R      = R         # circle radius
        self.D      = D         # Diffusion coefficient  [m/s^2]
        self.dt     = dt        # Time step              [s]
        self.steps  = steps     # Number of time steps   

        self.InitializeSimulation()

    def InitializeSimulation(self):
        # Initialize particle positions (r, theta, z)
        self.r = np.sqrt(np.random.uniform(0, 1, self.N)) * self.R
        self.theta = np.random.uniform(0, 2*np.pi, self.N)

        # Convert to Cartesian coordinates
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)

        # current position and trajectory container
        self.currPos = np.array([self.x,self.y])  # shape: (3, N)
        self.TRAJ = [self.currPos.copy()]  # store each timestep's positions

    def SingleStep(self):
        """
        Perform a single diffusion step for all particles.
        """
        dx = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)
        dy = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)

        # Update positions
        self.currPos[0] += dx
        self.currPos[1] += dy

        # Convert to cylindrical for boundary checks
        self.r = np.sqrt(self.currPos[0]**2 + self.currPos[1]**2)
        self.theta = np.arctan2(self.currPos[1], self.currPos[0])

        # Reflective boundary at r = R
        outside = self.r > self.R
        self.r[outside] = 2*self.R - self.r[outside]

        # Convert back to Cartesian
        self.currPos[0] = self.r * np.cos(self.theta)
        self.currPos[1] = self.r * np.sin(self.theta)

        # Save positions at this time step
        self.TRAJ.append(self.currPos.copy())

    def runSimulation(self):
        """
        Run the full simulation for 'steps' time steps.
        """
        for _ in range(self.steps):
            self.SingleStep()
        
    def plotTrajectory(self, fig, ax, particle_index=0):
        """
        Plot the 3D trajectory for a single particle.
        By default, plots the first particle (index 0).
        """
        # Convert list of arrays to one NumPy array of shape (steps+1, 3, N)
        traj_array = np.array(self.TRAJ)

        # Extract x, y,for the chosen particle
        x_vals = traj_array[:, 0, particle_index]
        y_vals = traj_array[:, 1, particle_index]

        # Plot the trajectory
        ax.plot(x_vals, y_vals, label=f'Particle {particle_index}')


class LineRandomWalk():
    def __init__ (self, 
                  N,
                  dt,
                  steps,
                  D,
                  H):
        
        # Simulation parameters
        self.N      = N         # Number of particles (simultaneous runs)
        self.H      = H         # line length
        self.D      = D         # Diffusion coefficient  [m/s^2]
        self.dt     = dt        # Time step              [s]
        self.steps  = steps     # Number of time steps   

        self.InitializeSimulation()

    def InitializeSimulation(self):
        # Initialize particle positions (r, theta, z)
        self.z = np.random.uniform(0, self.H, self.N)

        # current position and trajectory container
        self.currPos = np.array([self.z])  # shape: (1, N)
        self.TRAJ = [self.currPos.copy()]  # store each timestep's positions

    def SingleStep(self):
        """
        Perform a single diffusion step for all particles.
        """
        dz = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)

        # Update positions
        self.currPos[0] += dz

        # Reflective boundaries at z = 0 and z = H
        below_zero = self.currPos[0] < 0
        above_H    = self.currPos[0] > self.H

        self.currPos[0][below_zero] = -self.currPos[0][below_zero]
        self.currPos[0][above_H]    = 2*self.H - self.currPos[0][above_H]

        # Save positions at this time step
        self.TRAJ.append(self.currPos.copy())

    def runSimulation(self):
        """
        Run the full simulation for 'steps' time steps.
        """
        for _ in range(self.steps):
            self.SingleStep()
        
    def plotTrajectory(self, fig,ax, particle_index=0):
        """
        Plot the 3D trajectory for a single particle.
        By default, plots the first particle (index 0).
        """
        # Convert list of arrays to one NumPy array of shape (steps+1, 3, N)
        traj_array = np.array(self.TRAJ)

        # Extract x, y, z for the chosen particle
        z_vals = traj_array[:, 0, particle_index]

        # Plot the trajectory
        ax.plot(z_vals, label=f'Particle {particle_index}')




