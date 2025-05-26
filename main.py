import matplotlib.pyplot as plt
from hdr.oop_diffusion import *
from hdr.utils import *
from hdr.analytical_solution import *



if __name__ == "__main__":
    # ------------------------------------------------------------------------------------
    # CONSTANTS
    # ------------------------------------------------------------------------------------


    N = 10                                        # number of simultaneous simulations
    dt = 0.01                                    # time step for spatial advancement
    steps = 1000                                # total time steps, Time_arrya - 
    D = 0.0001                                   # Diffusion Coefficient [\um/s^2]
    R = 0.1                                      # Cylinder Radius       [\um ]
    H = 0.5                                      # Cylinder Height       [\um ]

    # ------------------------------------------------------------------------------------
    # EXPERIMENTAL
    # ------------------------------------------------------------------------------------


    #sim = CylinderRandomWalk(N, dt, steps, D, R, H)
    sim = CircleRandomWalk(N, dt, steps, D, R)
    #sim = LineRandomWalk(N, dt, steps, D, R)

    sim.runSimulation()                          # run the simulation
    sim.getMeanSquaredDisplacement()             # get the MSD 

    # ------------------------------------------------------------------------------------
    # ANALYTICAL
    # ------------------------------------------------------------------------------------
    t_array = np.linspace(0, dt * steps, steps)

    #ana = LineAnalyticalMSD(D, L, t_array)
    ana = CircleAnalyticalMSD(D, R, t_array)
    #ana = CylinderAnalyticalMSD(D, R, L, t_array)

    # ------------------------------------------------------------------------------------
    # ERROR ANALYSIS
    # ------------------------------------------------------------------------------------
    # ideally this should be comparing the experimental to the analytical

    # ------------------------------------------------------------------------------------
    # PREPARE FOR PLOTTING
    # ------------------------------------------------------------------------------------
    # self explanatory.


    # ------------------------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------------------------
    print(np.shape(np.mean(sim.MSD, axis=1)))

    fig = plt.figure(figsize=(12,18))

    # CYLINDER
    ax = fig.add_subplot(121, projection='3d')
    axMSD = fig.add_subplot(122)
    #axMeanMSD = fig.add_subplot(133)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Random Walk in a Cylinder') 
    Xc,Yc,Zc = data_for_cylinder_along_z(0,0,R,H)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2)

    # CIRCLE
    # ax = fig.add_subplot(121)
    # axMSD = fig.add_subplot(122)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Random Walk on a Circle')

    # LINE
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.set_title('Random Walk on a Line')

    # plotting
    for i in range(N):
        sim.plotTrajectory(fig, ax, particle_index=i)
        #sim.plotMSD(fig, axMSD, particle_index=i)

    sim.plotMeanMSD(fig, axMSD)
    axMSD.plot(t_array, ana, label=r'Analytical: $\langle \mathbf{r}^2(t) \rangle_c$')

    ax.legend()
    axMSD.legend()


    ax.grid()
    axMSD.grid()
    plt.show()
        
