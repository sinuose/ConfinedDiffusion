import os
import json
from datetime import datetime
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.constants import Boltzmann as kb
from hdr.oop_diffusion import *
from hdr.utils import *
from hdr.analytical_solution import *

import warnings



if __name__ == "__main__":
    # ------------------------------------------------------------------------------------
    # Retain each run in a safe place.
    # ------------------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simulation_output/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------
    # CONSTANTS
    # ------------------------------------------------------------------------------------


    N = 1000                                      # number of simultaneous simulations
    dt = 0.01                                    # time step for spatial advancement
    steps = 500                                 # total time steps, Time_array
    D = 10 * 10**(-12)                            # Diffusion Coefficient [m^2/s]
    R = 3 * 10**(-6)                             # Cylinder Radius       [m]
    H = 3 * 10**(-6)                             # Cylinder Height       [m]

    # Can also calculate D from Stokes Einstein for Experimental Comparison
    #T   = 22 + 273.15                       # K
    #R_h = 50//2 * 10**(-9)                  # m
    #D = kb * T / (6 * np.pi * 0.001 * R_h)  # m^2/s

    print(f"Simulation Settings: \n >Particles {N} \n >Radius {R:.2e} \n >Height {H:.2e} \n >Diffusion {D:.2e}")

    # Save parameters
    params = {
        "N": N,
        "dt": dt,
        "steps": steps,
        "D": D,
        "R": R,
        "H": H
    }
    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=4)

    # ------------------------------------------------------------------------------------
    # Define a Function for Each Figure to Keep the workspace clean
    # ------------------------------------------------------------------------------------
   
    # Helper to save MSD data
    def save_msd_data(name, t, msd):
        np.savetxt(os.path.join(output_dir, f"{name}_msd.csv"),
                np.column_stack((t, msd)), delimiter=",", header="Time,MSD", comments='')

    def Figure1():
        sims = {}

        # Cylinder
        sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
        sim_cyl.runSimulation()
        sim_cyl.getMeanSquaredDisplacement()
        t_array_cyl = np.linspace(0, dt * steps, steps)
        ana_cyl = CylinderAnalyticalMSDGPT(D, R, H, t_array_cyl)
        sims['cylinder'] = (sim_cyl, t_array_cyl, ana_cyl)

        # Circle
        sim_circ = CircleRandomWalk(N, dt, steps, D, R)
        sim_circ.runSimulation()
        sim_circ.getMeanSquaredDisplacement()
        t_array_circ = np.linspace(0, dt * steps, steps)
        ana_circ = CircleAnalyticalMSD(D, R, t_array_circ)
        sims['circle'] = (sim_circ, t_array_circ, ana_circ)

        # Line
        sim_line = LineRandomWalk(N, dt, steps, D, H)
        sim_line.runSimulation()
        sim_line.getMeanSquaredDisplacement()
        t_array_line = np.linspace(0, dt * steps, steps)
        ana_line = LineAnalyticalMSD(D, H, t_array_line)
        sims['line'] = (sim_line, t_array_line, ana_line)

        # ------------------------------------------------------------------------------------
        # ERROR ANALYSIS
        # ------------------------------------------------------------------------------------
        # ideally this should be comparing the experimental to the analytical

        # ------------------------------------------------------------------------------------
        # PLOTTING
        # ------------------------------------------------------------------------------------
        fig, axs = plt.subplots(3, 1, figsize=(6, 10), constrained_layout=True)

        labelsize = 10
        titlesize = 12
        legendsize = 8

        
        # -------------------------------------
        # CYLINDER MSD PLOT
        # -------------------------------------
        ax_cyl = axs[2]
        sim_cyl, t_cyl, ana_cyl = sims['cylinder']
        sim_cyl.plotMeanMSD(fig, ax_cyl)
        ax_cyl.plot(t_cyl, ana_cyl, linestyle='--', color='black', label="Analytical")
        save_msd_data("cylinder", t_cyl, ana_cyl)

        ax_cyl.set_title("Cylinder MSD", fontsize=titlesize)
        ax_cyl.set_xlabel("Time", fontsize=labelsize)
        ax_cyl.set_ylabel("MSD", fontsize=labelsize)
        ax_cyl.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax_cyl.legend(fontsize=legendsize, loc='lower right', frameon=False)

        # Inset: Cylinder Trajectory
        inset_position = [0.75, 0.122 , 0.2, 0.2]  # [left, bottom, width, height] in figure coordinates
        inset_cyl = fig.add_axes(inset_position, projection='3d')
        inset_cyl.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        inset_cyl.view_init(elev=30, azim=45)  # Isometric view

        Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, R, H)
        inset_cyl.plot_surface(Xc, Yc, Zc, alpha=0.15, color='gray', edgecolor='none')

        for i in range(1):
            sim_cyl.plotTrajectory(fig, inset_cyl, particle_index=i)

        inset_cyl.set_xticks([])
        inset_cyl.set_yticks([])
        inset_cyl.set_zticks([])
        inset_cyl.set_box_aspect([1, 1, 0.5])  # vertical cylinder
        inset_cyl.view_init(elev=20, azim=45)
        inset_cyl.grid(False)
        inset_cyl.set_facecolor((0, 0, 0, 0))  # transparent background

        # -------------------------------------
        # CIRCLE MSD PLOT
        # -------------------------------------
        ax_circ = axs[1]
        sim_circ, t_circ, ana_circ = sims['circle']
        sim_circ.plotMeanMSD(fig, ax_circ)
        ax_circ.plot(t_circ, ana_circ, linestyle='--', color='black', label="Analytical")
        save_msd_data("circle", t_circ, ana_circ)

        ax_circ.set_title("Circle MSD", fontsize=titlesize)
        ax_circ.set_xlabel("Time", fontsize=labelsize)
        ax_circ.set_ylabel("MSD", fontsize=labelsize)
        ax_circ.grid(True, linestyle=':', linewidth=0.5)
        ax_circ.legend(fontsize=legendsize, loc='lower right', frameon=False)

        # Inset: Circle Trajectory (2D)
        inset_circ = inset_axes(ax_circ, width="30%", height="30%", loc='upper right', borderpad=1)
        inset_circ.set_aspect('equal')
        for i in range(1):
            sim_circ.plotTrajectory(fig, inset_circ, particle_index=i)
        inset_circ.set_xticks([])
        inset_circ.set_yticks([])
        inset_circ.set_facecolor((0, 0, 0, 0))

        # -------------------------------------
        # LINE MSD PLOT
        # -------------------------------------
        ax_line = axs[0]
        sim_line, t_line, ana_line = sims['line']
        sim_line.plotMeanMSD(fig, ax_line)
        ax_line.plot(t_line, ana_line, linestyle='--', color='black', label="Analytical")
        save_msd_data("line", t_line, ana_line)

        ax_line.set_title("Line MSD", fontsize=titlesize)
        ax_line.set_xlabel("Time", fontsize=labelsize)
        ax_line.set_ylabel("MSD", fontsize=labelsize)
        ax_line.grid(True, linestyle=':', linewidth=0.5)
        ax_line.legend(fontsize=legendsize, loc='lower right', frameon=False)

        # Inset: Line Trajectory (1D)
        inset_line = inset_axes(ax_line, width="30%", height="30%", loc='upper right', borderpad=1)
        for i in range(1):
            sim_line.plotTrajectory(fig, inset_line, particle_index=i)
        inset_line.set_xticks([])
        inset_line.set_yticks([])
        inset_line.set_facecolor((0, 0, 0, 0))

        # -------------------------------------
        # Save and Display
        # -------------------------------------
        # Save SVG for LaTeX
        svg_path = os.path.join(output_dir, f"Figure1_{timestamp}.svg")
        fig.savefig(svg_path, format='svg')

        plt.suptitle("Mean Squared Displacement for Confined Random Walks", fontsize=14)
        plt.show()

    def Figure2():
        # Do just cylinder simulation
        sims = {}

        # Cylinder
        sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
        sim_cyl.runSimulation()
        # Note the decoupled MSD
        sim_cyl.getMeanSquaredDisplacementDecoupled()
        # sim_cyl.MSD/MSD_z/MSD_xy
        t_array_cyl = np.linspace(0, dt * steps, steps)
        ana_cyl = CylinderAnalyticalMSDGPT(D, R, H, t_array_cyl)
        sims['cylinder'] = (sim_cyl, t_array_cyl, ana_cyl)        

        # then overlap the linear and circle all on one graph
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), constrained_layout=True)

        labelsize = 10
        titlesize = 12
        legendsize = 8
        # -------------------------------------
        # CYLINDER MSD PLOT
        # -------------------------------------
        ax_cyl = axs[0]
        sim_cyl, t_cyl, ana_cyl = sims['cylinder']
        # Assuming that the diffusino is anisotropic, I want to measure D_xy, D_z
        sim_cyl.plotMeanMSDDecoupled(fig, axs[0])
        sim_cyl.plotMeanMSDComparison(fig, axs[1])

        #ax_cyl.plot(t_cyl, ana_cyl, linestyle='--', color='black', label="Analytical")
        #save_msd_data("cylinder_Decoupled", t_cyl, ana_cyl)

        ax_cyl.set_title("Cylinder MSD", fontsize=titlesize)
        ax_cyl.set_xlabel("Time", fontsize=labelsize)
        ax_cyl.set_ylabel("MSD", fontsize=labelsize)
        ax_cyl.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax_cyl.legend(fontsize=legendsize, loc='lower right', frameon=False)

         # Save SVG for LaTeX
        svg_path = os.path.join(output_dir, f"Figure2_{timestamp}.svg")
        fig.savefig(svg_path, format='svg')
        plt.show()

    def Figure3():
        # Do just cylinder simulation
        sims = {}

        # Cylinder
        sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
        sim_cyl.runSimulation()
        # Note the decoupled MSD
        sim_cyl.getMeanSquaredDisplacementDecoupled()
        # sim_cyl.MSD/MSD_z/MSD_xy
        t_array_cyl = np.linspace(0, dt * steps, steps)
        ana_cyl = CylinderAnalyticalMSDGPT(D, R, H, t_array_cyl)
        sims['cylinder'] = (sim_cyl, t_array_cyl, ana_cyl)
        
        fig, ax = plt.subplots(1,1)
         # -------------------------------------
        # CYLINDER MSD PLOT
        # -------------------------------------
        ax_cyl = ax
        sim_cyl, t_cyl, ana_cyl = sims['cylinder']
        #sim_cyl.plotMeanMSD(fig, ax_cyl)

        # Suppress warnings from curve fitting failures
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Define your model function
        def model(t, D, R, H):
            return CylinderAnalyticalMSDGPT(D, R, H, t)

        # Generate time array
        t_array = np.linspace(0, sim_cyl.dt * sim_cyl.steps, sim_cyl.MSD.shape[0])

        # Compute mean MSD across particles
        mean_msd = np.mean(sim_cyl.MSD, axis=1, dtype=np.float64)

        # Initialize lists for parameters
        D_array, a_array, L_array = [], [], []

        # Loop over individual MSDs (transpose to get shape: [particles, time])
        MSDs = sim_cyl.MSD.T  # Shape: (n_particles, n_timesteps)

        for msd in tqdm(MSDs[:], desc="Fitting MSD curves"):
            try:
                popt, _ = curve_fit(
                    model,
                    t_array,
                    msd,
                    p0=(1e-12, 1e-6, 1e-6),
                    maxfev=20000
                )
                D_array.append(popt[0])
                a_array.append(popt[1])
                L_array.append(popt[2])
            except Exception as e:
                continue  # Silently skip failures

        # Final fit on the mean MSD
        popt_mean, _ = curve_fit(
            model,
            t_array,
            mean_msd,
            p0=(1e-12, 1e-6, 1e-6),
            maxfev=20000
        )

        x = np.repeat(t_array, sim_cyl.MSD.shape[1])
        y = sim_cyl.MSD.flatten()

        # Create bins
        time_bins = np.linspace(t_array[0], t_array[-1], 200)
        msd_bins = np.linspace(np.min(sim_cyl.MSD), np.max(sim_cyl.MSD), 200)

        # Compute 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=[time_bins, msd_bins])

        # -------------------------------------
        # Plotting
        # -------------------------------------
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), constrained_layout=True)

        labelsize = 10
        titlesize = 12
        legendsize = 8

        # -------------------------------------
        # Top subplot: MSD distribution heatmap
        # -------------------------------------
        ax0 = axs[0]
        pcm = ax0.pcolormesh(
            xedges, yedges, hist.T,
            cmap='cubehelix_r',
            shading='auto',
            alpha=1
        )
        ax0.plot(t_array, mean_msd, color="blue", label="Mean MSD", linewidth=2)
        ax0.plot(t_array, model(t_array, *popt_mean), 'r--', label=f"Fit: D={popt_mean[0]:.2e} m²/s")

        ax0.set_title("Cylinder MSD Distribution", fontsize=titlesize)
        ax0.set_xlabel("Time [s]", fontsize=labelsize)
        ax0.set_ylabel("MSD [m²]", fontsize=labelsize)
        ax0.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax0.legend(fontsize=legendsize, loc='lower right', frameon=False)
        ax0.set_ylim(0, 1.5e-11)
        fig.colorbar(pcm, ax=ax0, label='Density')

        # -------------------------------------
        # Bottom subplot: Histogram of D values
        # -------------------------------------
        ax1 = axs[1]

        # Compute stats
        D_array = np.array(D_array)
        D_mean = np.mean(D_array)
        D_std = np.std(D_array)
        D_var = np.var(D_array)
        D_true = D 

        # Plot histogram
        ax1.hist(D_array, bins=30, color='slateblue', edgecolor='black', alpha=0.8, label='Fitted D values')

        # Add vertical lines for true and fitted mean
        ax1.axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D = {D_true:.2e}')
        ax1.axvline(D_mean, color='orange', linestyle='-', linewidth=2, label=f'Mean D = {D_mean:.2e}')
        ax1.axvspan(D_mean - D_std, D_mean + D_std, color='orange', alpha=0.2, label=f'±1σ = {D_std:.1e}')

        # Labels and title
        ax1.set_title("Distribution of Fitted D Values", fontsize=titlesize)
        ax1.set_xlabel("Diffusion Coefficient D [m²/s]", fontsize=labelsize)
        ax1.set_ylabel("Count", fontsize=labelsize)
        ax1.set_xlim(0, 3e-11)
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

        # Annotate key stats
        textstr = '\n'.join((
            f'N = {len(D_array)}',
            f'Mean = {D_mean:.2e}',
            f'Std Dev = {D_std:.1e}',
            f'Var = {D_var:.1e}'
        ))
        ax1.text(0.97, 0.95, textstr, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        # Legend
        ax1.legend(fontsize=legendsize, loc='upper left', frameon=False)

        # -------------------------------------
        # Save to SVG
        # -------------------------------------
        output_dir = "simulation_output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        svg_path = os.path.join(output_dir, f"Figure3_{timestamp}.svg")
        fig.savefig(svg_path, format='svg')

        plt.show()

    def Figure4():
        # Do just cylinder simulation
        sims = {}

        # Cylinder
        sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
        sim_cyl.runSimulation()
        # Note the decoupled MSD
        sim_cyl.getMeanSquaredDisplacementDecoupled()
        # sim_cyl.MSD/MSD_z/MSD_xy
        t_array_cyl = np.linspace(0, dt * steps, steps)
        #ana_cyl = CylinderAnalyticalMSDGPT(D, R, H, t_array_cyl)
        #sims['cylinder'] = (sim_cyl, t_array_cyl, ana_cyl)
        
        fig, ax = plt.subplots(1,1)
         # -------------------------------------
        # CYLINDER MSD PLOT
        # -------------------------------------
        ax_cyl = ax
        #sim_cyl, t_cyl, ana_cyl = sims['cylinder']
        #sim_cyl.plotMeanMSD(fig, ax_cyl)

        # Suppress warnings from curve fitting failures
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Define your model function
        def model(t, D, R):
            return CircleAnalyticalMSD(D, R, t)

        # Generate time array
        t_array = np.linspace(0, sim_cyl.dt * sim_cyl.steps, sim_cyl.MSD_xy.shape[0])

        # Compute mean MSD across particles
        mean_msd = np.mean(sim_cyl.MSD_xy, axis=1, dtype=np.float64)

        # Initialize lists for parameters
        D_array, a_array = [], []

        # Loop over individual MSDs (transpose to get shape: [particles, time])
        MSDs = sim_cyl.MSD_xy.T  # Shape: (n_particles, n_timesteps)

        for msd in tqdm(MSDs[:], desc="Fitting MSD curves"):
            try:
                popt, _ = curve_fit(
                    model,
                    t_array,
                    msd,
                    p0=(1e-12, 1e-6),
                    maxfev=20000
                )
                D_array.append(popt[0])
                a_array.append(popt[1])
            except Exception as e:
                continue  # Silently skip failures

        # Final fit on the mean MSD
        popt_mean, _ = curve_fit(
            model,
            t_array,
            mean_msd,
            p0=(1e-12, 1e-6),
            maxfev=20000
        )

        x = np.repeat(t_array, sim_cyl.MSD_xy.shape[1])
        y = sim_cyl.MSD_xy.flatten()

        # Create bins
        time_bins = np.linspace(t_array[0], t_array[-1], 200)
        msd_bins = np.linspace(np.min(sim_cyl.MSD_xy), np.max(sim_cyl.MSD_xy), 200)

        # Compute 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=[time_bins, msd_bins])

        # -------------------------------------
        # Plotting
        # -------------------------------------
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), constrained_layout=True)

        labelsize = 10
        titlesize = 12
        legendsize = 8

        # -------------------------------------
        # Top subplot: MSD distribution heatmap
        # -------------------------------------
        ax0 = axs[0]
        pcm = ax0.pcolormesh(
            xedges, yedges, hist.T,
            cmap='cubehelix_r',
            shading='auto',
            alpha=1
        )
        ax0.plot(t_array, mean_msd, color="blue", label="Mean MSD", linewidth=2)
        ax0.plot(t_array, model(t_array, *popt_mean), 'r--', label=f"Fit: D={popt_mean[0]:.2e} m²/s")

        ax0.set_title("Cylinder Polar MSD Distribution", fontsize=titlesize)
        ax0.set_xlabel("Time [s]", fontsize=labelsize)
        ax0.set_ylabel("MSD [m²]", fontsize=labelsize)
        ax0.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax0.legend(fontsize=legendsize, loc='lower right', frameon=False)
        ax0.set_ylim(0, 1.4e-11)
        fig.colorbar(pcm, ax=ax0, label='Density')

        # -------------------------------------
        # Bottom subplot: Histogram of D values
        # -------------------------------------
        ax1 = axs[1]

        # Compute stats
        D_array = np.array(D_array)
        D_mean = np.mean(D_array)
        D_std = np.std(D_array)
        D_var = np.var(D_array)
        D_true = D 

        # Plot histogram
        ax1.hist(D_array, bins=30, color='slateblue', edgecolor='black', alpha=0.8, label='Fitted D values')

        # Add vertical lines for true and fitted mean
        ax1.axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D = {D_true:.2e}')
        ax1.axvline(D_mean, color='orange', linestyle='-', linewidth=2, label=f'Mean D = {D_mean:.2e}')
        ax1.axvspan(D_mean - D_std, D_mean + D_std, color='orange', alpha=0.2, label=f'±1σ = {D_std:.1e}')

        # Labels and title
        ax1.set_title("Distribution of Fitted D Values", fontsize=titlesize)
        ax1.set_xlabel("Diffusion Coefficient D [m²/s]", fontsize=labelsize)
        ax1.set_ylabel("Count", fontsize=labelsize)
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

        # Annotate key stats
        textstr = '\n'.join((
            f'N = {len(D_array)}',
            f'Mean = {D_mean:.2e}',
            f'Std Dev = {D_std:.1e}',
            f'Var = {D_var:.1e}'
        ))
        ax1.text(0.97, 0.95, textstr, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        # Legend
        ax1.legend(fontsize=legendsize, loc='upper left', frameon=False)

        # -------------------------------------
        # Save to SVG
        # -------------------------------------
        svg_path = os.path.join(output_dir, f"Figure4_{timestamp}.svg")
        fig.savefig(svg_path, format='svg')

        plt.show()

    def Figure5():
        # -------------------------------------
        # Parameters (update these as needed)
        # -------------------------------------
        sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
        sim_cyl.runSimulation()
        t_array_cyl = np.linspace(0, dt * steps, steps)

        sim_cyl.getDiffusionMLE()
        D_MLE = np.array(sim_cyl.D_MLE)
        D_true = D
        D_mean = np.mean(D_MLE)
        D_std = np.std(D_MLE)
        D_var = np.var(D_MLE)

        # -------------------------------------
        # Plot setup
        # -------------------------------------
        fig, ax = plt.subplots(figsize=(12, 8), )
        labelsize = 11
        titlesize = 13
        legendsize = 9

        # Histogram
        ax.hist(D_MLE, bins=30, color='steelblue', edgecolor='black', alpha=0.8, label='MLE Estimated D')

        # Lines: true D and mean
        ax.axvline(D_true, color='red', linestyle='--', linewidth=2, label=f"True D = {D_true:.2e}")
        ax.axvline(D_mean, color='orange', linestyle='-', linewidth=2, label=f"Mean D = {D_mean:.2e}")
        ax.axvspan(D_mean - D_std, D_mean + D_std, color='orange', alpha=0.2, label=f"±1σ = {D_std:.1e}")

        # Labels and style
        ax.set_title("Histogram of MLE Estimated D", fontsize=titlesize)
        ax.set_xlabel("Diffusion Coefficient D [m²/s]", fontsize=labelsize)
        ax.set_ylabel("Count", fontsize=labelsize)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.legend(fontsize=legendsize, loc='upper left', frameon=False)

        # Stats box
        textstr = '\n'.join((
            f'N = {len(D_MLE)}',
            f'Mean = {D_mean:.2e}',
            f'Std Dev = {D_std:.1e}',
            f'Var = {D_var:.1e}'
        ))
        ax.text(0.97, 0.95, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        # Save as SVG
        svg_path = os.path.join(output_dir, f"Figure5_{timestamp}.svg")
        fig.savefig(svg_path, format='svg')

        plt.tight_layout()
        plt.show()



    # VALIDATION OF THE CYLINDEIRCAL MODEL
    #Figure1()
    
    # COMPARING THE CIRCULAR MSD TO THE CYLINDERICAL MSD
    #Figure2()

    # EXTRACTING DIFFUSION FROM MSD AND COMPARING HOW CIRCULAR/LINEAR/CYLYNDERICAL GET THE NUMBERS
    #Figure3()

    #Figure4()

    # DIFFUSION BEST ESTIMATOR
    Figure5()


    # -----
    # END OF PROGRAM
    # ------