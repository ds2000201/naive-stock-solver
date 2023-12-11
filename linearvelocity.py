# Code for Linear Velocity
import numpy as np
import matplotlib.pyplot as plt

# Constants
Gamma = 5  # Thermal conductivity
phi_x0 = 100.0  # Boundary condition at x=0
phi_xL = 0.0  # Boundary condition at x=L
phi_y0 = 0.0  # Boundary condition at y=0
phi_yL = 100.0  # Boundary condition at y=L
u_x = 2  # x-velocity
u_y = 2  # y-velocity
den = 1  # density

# Grid parameters
Lx = 1.0  # Length of the domain in the x-direction
Ly = 1.0  # Length of the domain in the y-direction


def solve(Nx, Ny):
    dx = Lx / Nx  # Grid spacing in the x-direction
    dy = Ly / Ny  # Grid spacing in the y-direction

    # Initializing solution Matrix
    phi = np.zeros((Nx, Ny))

    # Jacobi iterative solver
    max_iterations = 1000
    tolerance = 1e-4

    scheme = input("Scheme for discretization of convective term (Enter 1 for UDS or any other button for CDS): ")
    # Upwind Difference Scheme
    if scheme == '1':
        # Case when gamma is not zero
        if Gamma != 0:
            for iteration in range(max_iterations):
                old_phi = phi.copy()
                # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = ((Gamma * dy / dx + den * u_x) * phi[i - 1, j] + (Gamma * dy / dx) * phi[
                            i + 1, j] + (Gamma * dx / dy + den * u_y) * phi[i, j - 1] + (Gamma * dx / dy) * phi[
                                         i, j + 1]) / (
                                                Gamma * (dy / dx + dy / dx + dx / dy + dx / dy) + den * (u_x + u_y))

                # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = ((Gamma * dy / dx + den * u_x) * phi[i - 1, 0] + (Gamma * dy / dx) * phi[i + 1, 0] + (
                                2 * Gamma * dx / dy + den * u_y) * phi_y0 + (Gamma * dx / dy) * phi[i, 1]) / (
                                            Gamma * (dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x + u_y))

                # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny - 1] = ((Gamma * dy / dx + den * u_x) * phi[i - 1, Ny - 1] + (Gamma * dy / dx) * phi[
                        i + 1, Ny - 1] + (Gamma * dx / dy + den * u_y) * phi[i, Ny - 2] + (
                                                  2 * Gamma * dx / dy) * phi_yL - den * u_y) / (
                                                 Gamma * (dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x))

                # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx - 1, j] = ((Gamma * dy / dx + den * u_x) * phi[Nx - 1, j] + (
                                2 * Gamma * dy / dx - den * u_x) * phi_xL + (Gamma * dx / dy + den * u_y) * phi[
                                          Nx - 1, j - 1] + (Gamma * dx / dy) * phi[Nx - 1, j + 1]) / (
                                                 Gamma * (dy / dx + 2 * dy / dx + dx / dy + dx / dy) + den * (u_y))

                # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx) * phi[1, j] + (
                                Gamma * dx / dy + den * u_y) * phi[0, j - 1] + (Gamma * dx / dy) * phi[0, j + 1]) / (
                                            Gamma * (2 * dy / dx + dy / dx + dx / dy + dx / dy) + den * (u_x + u_y))

                # North-eastern corner
                phi[Nx - 1, Ny - 1] = ((Gamma * dy / dx + den * u_x) * phi[Nx - 1, Ny - 1] + (
                            2 * Gamma * dy / dx - den * u_x) * phi_xL + (Gamma * dx / dy + den * u_y) * phi[
                                           Nx - 1, Ny - 2] + (2 * Gamma * dx / dy) * phi_yL - den * u_y) / (
                                                  Gamma * (dy / dx + 2 * dy / dx + 2 * dx / dy + dx / dy))

                # South-eastern corner
                phi[Nx - 1, 0] = ((Gamma * dy / dx + den * u_x) * phi[Nx - 2, 0] + (
                            2 * Gamma * dy / dx - den * u_x) * phi_xL + (2 * Gamma * dx / dy + den * u_y) * phi_y0 + (
                                              Gamma * dx / dy) * phi[Nx - 1, 1]) / (
                                             Gamma * (dy / dx + 2 * dy / dx + 2 * dx / dy + dx / dy) + den * (u_x))

                # North-western corner
                phi[0, Ny - 1] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx) * phi[1, Ny - 1] + (
                            Gamma * dx / dy + den * u_y) * phi[0, Ny - 2] + (
                                              2 * Gamma * dx / dy) * phi_yL - den * u_y) / (
                                             Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x))

                # South-western corner
                phi[0, 0] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx) * phi[1, 0] + (
                            2 * Gamma * dx / dy + den * u_y) * phi_y0 + (Gamma * dx / dy) * phi[0, 1]) / (
                                        Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x + u_y))

                sum = 0
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sum += (phi[i, j] - old_phi[i, j]) ** 2
                er = np.sqrt(sum / (Nx * Ny))

                print(f"Iteration # {iteration} and max error is {er}")
                print(f"NW= {phi[0, Ny - 1]} NE={phi[Nx - 1, Ny - 1]} and max error is {er}")
                # Check for convergence
                if er < tolerance:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

        # Case where Gamma = 0
        else:
            for iteration in range(max_iterations):
                old_phi = phi.copy()
                # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = (u_x * phi[i - 1, j] + u_y * phi[i, j - 1]) / (u_x + u_y)

                # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = (u_x * phi[i - 1, 0] + u_y * phi_y0) / (u_x + u_y)

                # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny - 1] = (u_x * phi[i - 1, Ny - 1] + u_y * phi[i, Ny - 2] - u_y * phi_yL) / u_x

                # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx - 1, j] = (u_x * phi[Nx - 2, j] + u_y * phi[Nx - 1, j - 1] - u_x * phi_xL) / u_y

                # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = (u_x * phi_x0 + u_y * phi[0, j - 1]) / (u_x + u_y)

                # North-eastern corner
                phi[Nx - 1, Ny - 1] = (u_x * phi_xL + u_y * phi_yL) / (u_x + u_y)

                # South-eastern corner
                phi[Nx - 1, 0] = (u_x * phi[Nx - 2, 0] + u_y * phi_y0 - u_x * phi_xL) / u_y

                # North-western corner
                phi[0, Ny - 1] = (u_x * phi_x0 + u_y * phi[0, Ny - 2] - u_y * phi_yL) / u_x

                # South-western corner
                phi[0, 0] = (u_x * phi_x0 + u_y * phi_y0) / (u_x + u_y)

                sum = 0
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sum += (phi[i, j] - old_phi[i, j]) ** 2
                er = np.sqrt(sum / (Nx * Ny))

                print(f"Iteration # {iteration} and max error is {er}")

                # Check for convergence
                if er < tolerance:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

    # Central Difference Scheme
    else:
        # Case when gamma is not zero
        if Gamma != 0:
            for iteration in range(max_iterations):
                old_phi = phi.copy()
                # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[i - 1, j] + (
                                    Gamma * dy / dx - 0.5 * den * u_x) * phi[i + 1, j] + (
                                             Gamma * dx / dy + 0.5 * den * u_y) * phi[i, j - 1] + (
                                                 Gamma * dx / dy - 0.5 * den * u_y) * phi[i, j + 1]) / (
                                            Gamma * (dy / dx + dy / dx + dx / dy + dx / dy))

                # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[i - 1, 0] + (
                                Gamma * dy / dx - 0.5 * den * u_x) * phi[i + 1, 0] + (
                                         2 * Gamma * dx / dy + den * u_y) * phi_y0 + (
                                             Gamma * dx / dy - 0.5 * den * u_y) * phi[i, 1]) / (
                                        Gamma * (dy / dx + dy / dx + dx / dy + 2 * dx / dy) - 0.5 * den * u_y)

                # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny - 1] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[i - 1, Ny - 1] + (
                                Gamma * dy / dx - 0.5 * den * u_x) * phi[i + 1, Ny - 1] + (
                                              Gamma * dx / dy + 0.5 * den * u_y) * phi[i, Ny - 2] + (
                                                  2 * Gamma * dx / dy - den * u_y) * phi_yL) / (
                                             Gamma * (dy / dx + dy / dx + 2 * dx / dy + dx / dy) - 0.5 * den * u_y)

                # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx - 1, j] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[Nx - 2, j] + (
                                2 * Gamma * dy / dx - den * u_x) * phi_xL + (
                                              Gamma * dx / dy + 0.5 * den * u_y) * phi[Nx - 1, j - 1] + (
                                                  Gamma * dx / dy - 0.5 * den * u_y) * phi[Nx - 1, j + 1]) / (
                                             Gamma * (dy / dx + 2 * dy / dx + dx / dy + dx / dy) - 0.5 * den * u_x)

                # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx - 0.5 * den * u_x) * phi[
                        1, j] + (
                                         Gamma * dx / dy + 0.5 * den * u_y) * phi[0, j - 1] + (
                                             Gamma * dx / dy - 0.5 * den * u_y) * phi[0, j + 1]) / (
                                        Gamma * (2 * dy / dx + dy / dx + dx / dy + dx / dy) - 0.5 * den * u_x)

                # North-eastern corner
                phi[Nx - 1, Ny - 1] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[Nx - 2, Ny - 1] + (
                            2 * Gamma * dy / dx - den * u_x) * phi_xL + (
                                               Gamma * dx / dy + 0.5 * den * u_y) * phi[Nx - 1, Ny - 2] + (
                                                   2 * Gamma * dx / dy - den * u_y) * phi_yL) / (
                                              Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) - 0.5 * den * (
                                                  u_x + u_y))

                # South-eastern corner
                phi[Nx - 1, 0] = ((Gamma * dy / dx + 0.5 * den * u_x) * phi[Nx - 2, 0] + (
                            2 * Gamma * dy / dx - den * u_x) * phi_xL + (
                                          2 * Gamma * dx / dy + den * u_y) * phi_y0 + (
                                              Gamma * dx / dy - 0.5 * den * u_y) * phi[Nx - 1, 1]) / (
                                         Gamma * (dy / dx + 2 * dy / dx + 2 * dx / dy + dx / dy) - 0.5 * den * (
                                             u_x + u_y))

                # North-western corner
                phi[0, Ny - 1] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx) * phi[1, Ny - 1] + (
                        Gamma * dx / dy + den * u_y) * phi[0, Ny - 2] + (2 * Gamma * dx / dy) * phi_yL - den * u_y) / (
                                         Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x + u_y))

                # South-western corner
                phi[0, 0] = ((2 * Gamma * dy / dx + den * u_x) * phi_x0 + (Gamma * dy / dx - 0.5 * den * u_x) * phi[
                    1, 0] + (
                                     2 * Gamma * dx / dy + den * u_y) * phi_y0 + (Gamma * dx / dy - 0.5 * den * u_y) *
                             phi[0, 1]) / (
                                    Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) - 0.5 * den * (u_x + u_y))

                sum = 0
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sum += (phi[i, j] - old_phi[i, j]) ** 2
                er = np.sqrt(sum / (Nx * Ny))

                print(f"Iteration # {iteration} and max error is {er}")
                print(f"NW= {phi[0, Ny - 1]} NE={phi[Nx - 1, Ny - 1]} and max error is {er}")
                # Check for convergence
                if er < tolerance:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

        # Case where Gamma = 0
        else:
            print("CDS not possible for Gamma=0")
    return phi


p = input("enter mesh size: ")
p = int(p)
phi = solve(p, p)

# Plotting the phi values along the other diagonal
other_diagonal_values = [phi[i, p - 1 - i] for i in range(min(p, p))]
plt.plot(other_diagonal_values, color='red', linestyle='dashed', label='100x100')

plt.legend()
plt.show()

# Plotting the solution
x = np.linspace(0, Lx, 100)
y = np.linspace(0, Ly, 100)
X, Y = np.meshgrid(x, y)
plt.contourf(Y, X, phi, cmap='jet', levels=150)
plt.colorbar(label='Temperature (phi)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


