import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import inv
matplotlib.use('Qt5Agg')

from scipy.optimize import curve_fit
#  --------------------------------------------------------------------------------
# MAIN CLASS
#  --------------------------------------------------------------------------------


#  --------------------------------------------------------------------------------
# CYLINDER
#  --------------------------------------------------------------------------------

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
        for _ in tqdm(range(self.steps)):
            self.SingleStep()

    def getMeanSquaredDisplacement(self):
        """
        Computes MSD_i for each particle j as:
        MSD_i(j) = Sum_{n} [(x_{n+i,j} - x_{n,j})^2 + (y_{n+i,j} - y_{n,j})^2]
        over all valid n, for i = 0..steps, returning a matrix of shape (steps+1, N).
        """
        # Convert list of arrays to NumPy array of shape (T, 3, N), where T = steps+1
        traj_array = np.array(self.TRAJ)
        T, _, N = traj_array.shape

        msd_matrix = np.zeros((T, N))

        # Loop over lag time i (from 0 to T-1)
        for n in range(T):
            displacements = traj_array[n:, :3, :] - traj_array[:T - n, :3, :]
            # displacements shape: (T-i, 2, N)
            squared_disp = np.sum(displacements**2, axis=1)  # shape: (T-n, N)

            msd_matrix[n] = np.mean(squared_disp, axis=0)   # 1/(T-n) averaging
        
        self.MSD = msd_matrix  # Shape: (steps+1, N)


    def getMeanSquaredDisplacementDecoupled(self):
        """
        Computes MSD_i for each particle j as:
        MSD_i(j) = Sum_{n} [(x_{n+i,j} - x_{n,j})^2 + (y_{n+i,j} - y_{n,j})^2]
        over all valid n, for i = 0..steps, returning a matrix of shape (steps+1, N).

        This decouples the system into xy and z components which is intended to use to test the model
        """
        # Convert list of arrays to NumPy array of shape (T, 3, N), where T = steps+1
        traj_array = np.array(self.TRAJ)
        T, _, N = traj_array.shape

        msd_xy = np.zeros((T, N))
        msd_z = np.zeros((T, N))
        msd_total = np.zeros((T, N))

        # Loop over lag time i (from 0 to T-1)
        for n in range(T):
            displacements = traj_array[n:, :, :] - traj_array[:T - n, :, :]  # (T-n, 3, N)
            
            squared_disp_xy = displacements[:, 0, :]**2 + displacements[:, 1, :]**2
            squared_disp_z = displacements[:, 2, :]**2
            squared_disp_total = squared_disp_xy + squared_disp_z

            msd_xy[n] = np.mean(squared_disp_xy, axis=0)
            msd_z[n] = np.mean(squared_disp_z, axis=0)
            msd_total[n] = np.mean(squared_disp_total, axis=0)
        
        # Store results
        self.MSD_xy = msd_xy  # shape (T, N)
        self.MSD_z = msd_z
        self.MSD = msd_total  # Shape: (steps+1, N)

    def plotTrajectory(self, fig,ax, particle_index=0):
        """
        Plot the 3D trajectory for a single particle.
        By default, plots the first particle (index 0).
        """
        # Convert list of arrays to one NumPy array of shape (steps+1, 3, N)
        traj_array = np.array(self.TRAJ)

        # Extract x, y, z for the chosen particle
        x_vals = traj_array[:100, 0, particle_index]
        y_vals = traj_array[:100, 1, particle_index]
        z_vals = traj_array[:100, 2, particle_index]

        # Plot the trajectory
        ax.plot(x_vals, y_vals, z_vals, label=f'Particle {particle_index}')

    def plotMSD(self, fig, ax, particle_index=0):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, self.MSD[:, particle_index], alpha=0.3, label=f"Particle {particle_index}") 

        ax.set_title("MSD for Individual Particles")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()
    
    def plotMeanMSD(self, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, np.mean(self.MSD, axis=1), alpha=0.3, label="Mean MSD") 

        ax.set_title("Ensemble MSD")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()

    def plotMeanMSDDecoupled(self, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, np.mean(self.MSD, axis=1), alpha=0.3, label="Mean MSD") 
        ax.plot(time, np.mean(self.MSD_xy, axis=1), alpha=0.3, label="Mean MSD xy")
        ax.plot(time, np.mean(self.MSD_z, axis=1), alpha=0.3, label="Mean MSD z")  

        # plot a red adjustment using the exact formula:
        ax.plot(time, (1 + self.H**2 / (6 * self.R**2)) * np.mean(self.MSD_xy, axis=1), "r--", alpha=0.3, label="Adjusted XY")

        ax.set_title("Ensemble MSD")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()

    def plotMeanMSDComparison(self, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, np.mean(np.abs(self.MSD - self.MSD_xy), axis=1), alpha=0.3, label="$|<x_{cyl}> - <x_{circ}>|$") 
        ax.plot(time, np.ones(len(time)) * self.H**2 / 6, "r--", alpha=0.3, label="Axial MSD Limit")  

        ax.set_title("Comparison")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()

    def extractDiffusionFromMSD(self, model, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")


        # Generate time array
        t_array = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot mean MSD
        mean_msd = np.mean(self.MSD, axis=1)
        # extract diffusion for mean MSD first

    def compute_covariance_matrix(self, T):
        """Returns covariance matrix R for Brownian motion (1D, no noise)."""
        R = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1), (T, T), dtype=int)
        return R

    def getDiffusionMLE(self):
        """
        Computes the MLE diffusion coefficient for a 3D trajectory array.
        
        Parameters:
        - traj_array: np.ndarray of shape (T, 3, N)
        - dt: float, time step between trajectory points
        
        Returns:
        - D_mle: np.ndarray of shape (N,), estimated diffusion coefficients
        """
        traj_array = np.array(self.TRAJ)

        T, d, N = traj_array.shape
        R = self.compute_covariance_matrix(T)
        R_inv = inv(R)
        
        D_mle = np.zeros(N)

        for n in range(N):
            sum_proj = 0.0
            for dim in range(d):
                x = traj_array[:, dim, n]
                sum_proj += x @ R_inv @ x  # x^T R^{-1} x
            D_mle[n] = sum_proj / (2 * d * (T - 1) * self.dt)

        self.D_MLE = D_mle
        

#  --------------------------------------------------------------------------------
# CIRCLE
#  --------------------------------------------------------------------------------

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
        for _ in tqdm(range(self.steps)):
            self.SingleStep()
    
    def getMeanSquaredDisplacement(self):
        """
        Computes MSD_i for each particle j as:
        MSD_i(j) = Sum_{n} [(x_{n+i,j} - x_{n,j})^2 + (y_{n+i,j} - y_{n,j})^2]
        over all valid n, for i = 0..steps, returning a matrix of shape (steps+1, N).
        """
        # Convert list of arrays to NumPy array of shape (T, 3, N), where T = steps+1
        traj_array = np.array(self.TRAJ)
        T, _, N = traj_array.shape

        msd_matrix = np.zeros((T, N))

        # Loop over lag time i (from 0 to T-1)
        for n in range(T):
            displacements = traj_array[n:, :2, :] - traj_array[:T - n, :2, :]
            # displacements shape: (T-i, 2, N)
            squared_disp = np.sum(displacements**2, axis=1)  # shape: (T-n, N)

            msd_matrix[n] = np.mean(squared_disp, axis=0)   # 1/(T-n) averaging
        
        self.MSD = msd_matrix  # Shape: (steps+1, N)
    
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

    def plotMSD(self, fig, ax, particle_index=0):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, self.MSD[:, particle_index], alpha=0.3, label=f"Particle {particle_index}") 

        ax.set_title("MSD for Individual Particles")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()

    def plotMeanMSD(self, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, np.mean(self.MSD, axis=1), alpha=0.3, label="Mean MSD") 

        ax.set_title("Ensemble MSD")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()


#  --------------------------------------------------------------------------------
#  LINE
#  --------------------------------------------------------------------------------

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
        for _ in tqdm(range(self.steps)):
            self.SingleStep()
    
    def getMeanSquaredDisplacement(self):
        """
        Computes MSD for each particle separately, returning a matrix of shape (steps+1, N).
        """
        # Convert list of arrays to one NumPy array of shape (steps+1, 3, N)
        traj_array = np.array(self.TRAJ)  # Shape: (steps+1, 3, N)

        # Reference initial positions for all particles
        x0 = traj_array[0, 0, :]

        # Compute squared displacement at each time step for each particle
        msd_matrix = (traj_array[:, 0, :] - x0)**2

        # Store as an attribute for later use
        self.MSD = msd_matrix  # Shape: (steps+1, N)

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

    def plotMSD(self, fig, ax, particle_index=0):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, self.MSD[:, particle_index], alpha=0.3, label=f"Particle {particle_index}") 

        ax.set_title("MSD for Individual Particles")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()

    def plotMeanMSD(self, fig, ax):
        """
        Plot the Mean Squared Displacement over time.
        Plot MSD curves for individual particles.
        """
        if not hasattr(self, 'MSD'):
            raise ValueError("MSD has not been computed. Run getMeanSquaredDisplacement() first.")

        # Generate time array
        time = np.linspace(0, self.dt * self.steps, self.MSD.shape[0])

        # Plot MSD for each individual particle
        ax.plot(time, np.mean(self.MSD, axis=1), alpha=0.3, label="Mean MSD") 

        ax.set_title("Ensemble MSD")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MSD [m^2]")
        ax.legend()
        ax.grid()





