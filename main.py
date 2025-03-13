import matplotlib.pyplot as plt
from hdr.oop_diffusion import *

if __name__ == "__main__":
    # parameters for the simulation
    N = 1
    dt = 0.01
    steps = 10000
    D = 1
    R = 10
    H = 20

    #sim = CylinderRandomWalk(N, dt, steps, D, R, H)
    #sim = CircleRandomWalk(N, dt, steps, D, R)
    sim = LineRandomWalk(N, dt, steps, D, R)
    sim.runSimulation()


    fig = plt.figure(figsize=(12,12))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    # plotting
    for i in range(N):
        sim.plotTrajectory(fig, ax, particle_index=i)

    # CYLINDER
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.set_title('Random Walk in a Cylinder') 

    # CIRCLE
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_title('Random Walk on a Circle')

    # LINE
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Random Walk on a Line')

    ax.legend()
    ax.grid()
    plt.show()
        
