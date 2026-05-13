import numpy as np

def mandelbrot(xmin=-2, xmax=2, ymin=-2, ymax=2, N=1000, max_iter=100):
    x = np.linspace(xmin, xmax, N)  # N values between xmin and xmax
    y = np.linspace(ymin, ymax, N)

    C = x[None, :] + 1j*y[:, None]  # creating full grid of complex numbers; x[None, :]->(1, N), y[:, None]->(N, 1)
    Z = np.zeros_like(C)  # setting z_0=0

    divergence_iter = np.zeros(C.shape, dtype=int)  # creating array to store the iteration number of divergent points
                                                    # entire array is filled with 0's initially
    bounded = np.ones(C.shape, dtype=bool)  # creating Boolean array to store whether each point is still considered bounded
                                            # every entry is True initially
    for i in range(1, max_iter + 1):  # initializing iteration loop
        Z[bounded] = Z[bounded]**2 + C[bounded]  # Mandelbrot iteration (for points still marked as bounded)

        diverged = np.abs(Z) > 2  # checking which points currently satisfy |z|>2
        newly_diverged = diverged & bounded  # points that just exceeded |z|>2, but were marked as bounded before this step

        divergence_iter[newly_diverged] = i  # for points diverging at iteration i, store that iteration number
        bounded[newly_diverged] = False  # marking those points as no longer bounded

    return x, y, bounded, divergence_iter  # returns Boolean array "bounded"; true means point did not diverge within max_iter
                                     # returns integer array "divergence_iter"; entries give iteration number where point diverged