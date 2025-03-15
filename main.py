import matplotlib.pyplot as plt
from hdr.oop_diffusion import *
from hdr.utils import *



if __name__ == "__main__":
    # parameters for the simulation
    N = 4
    dt = 0.01
    steps = 10000
    D = 0.0001
    R = 0.1
    H = 0.5

    sim = CylinderRandomWalk(N, dt, steps, D, R, H)
    #sim = CircleRandomWalk(N, dt, steps, D, R)
    #sim = LineRandomWalk(N, dt, steps, D, R)
    sim.runSimulation()
    sim.getMeanSquaredDisplacement()
    print(np.shape(sim.MSD))


    fig = plt.figure(figsize=(12,18))

    # CYLINDER
    ax = fig.add_subplot(121, projection='3d')
    axMSD = fig.add_subplot(122)

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
        sim.plotMSD(fig, axMSD, particle_index=i)

    ax.legend()
    axMSD.legend()

    ax.grid()
    axMSD.grid()
    plt.show()
        
