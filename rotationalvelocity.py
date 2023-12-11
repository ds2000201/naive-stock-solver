#Code for Rotational Velocity
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import gmres

# Constants
Gamma = 5  # Thermal conductivity
phi_x0 = 100.0  # Boundary condition at x=0
phi_xL = 0.0    # Boundary condition at x=L
phi_y0 = 0.0    # Boundary condition at y=0
phi_yL = 100.0  # Boundary condition at y=L
den = 1         # Density
max_iterations = 100  # Maximum number of iterations for convergence
tolerance = 1e-6       # Tolerance for convergence

# Grid parameters
Lx = 1.0  # Length of the domain in the x-direction
Ly = 1.0  # Length of the domain in the y-direction
scheme = input("Scheme for discretization of convective term (Enter 1 for UDS or any other button for CDS): ")

def func(Nx, Ny):
    dx = Lx / Nx  # Grid spacing in the x-direction
    dy = Ly / Ny  # Grid spacing in the y-direction
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    # Velocity field
    X, Y = np.meshgrid(x - 0.5 * Lx, y - 0.5 * Ly)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    u_x = -r * np.sin(theta)
    u_y = r * np.cos(theta)

    phi = np.zeros((Nx, Ny))

    # Initialize sparse matrix A if required
    A = lil_matrix((Nx * Ny, Nx * Ny))
    # b = np.zeros(Nx * Ny)  # Initialize b as a zero vector
    b = np.ones(Nx * Ny)  # Initialize b as a one vector

    # Boundary conditions
    phi[:, 0] = phi_y0   # South
    phi[:, -1] = phi_yL  # North
    phi[0, :] = phi_x0   # West
    phi[-1, :] = phi_xL  # East

    # Upwind Difference Scheme
    if scheme == '1':
        #Case when gamma is not zero
        if Gamma!=0:
            for iteration in range(max_iterations):
                old_phi = phi.copy()
            # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = ((Gamma*dy/dx+den*u_x[i,j])*phi[i-1,j]+ (Gamma*dy/dx)*phi[i+1,j]+(Gamma*dx/dy+den*u_y[i,j])*phi[i,j-1]+(Gamma*dx/dy)*phi[i,j+1])/(Gamma * (dy/dx + dy/dx + dx/dy + dx/dy) + den*(u_x[i,j]+u_y[i,j]))

            # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = ((Gamma*dy/dx+den*u_x[i,0])*phi[i-1,0]+ (Gamma*dy/dx)*phi[i+1,0]+(2*Gamma*dx/dy+den*u_y[i,0])*phi_y0+(Gamma*dx/dy)*phi[i,1])/(Gamma * (dy/dx + dy/dx + 2*dx/dy + dx/dy) + den*(u_x[i,0]+u_y[i,0]))

            # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny-1] = ((Gamma*dy/dx+den*u_x[i,Ny-1])*phi[i-1,Ny-1]+ (Gamma*dy/dx)*phi[i+1,Ny-1]+(Gamma*dx/dy+den*u_y[i,Ny-1])*phi[i,Ny-2]+(2*Gamma*dx/dy)*phi_yL-den*u_y[i,Ny-1])/(Gamma * (dy/dx + dy/dx + 2*dx/dy + dx/dy) + den*(u_x[i,Ny-1]))

            # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx-1,j] = ((Gamma*dy/dx+den*u_x[Nx-1,j])*phi[Nx-1,j]+ (2*Gamma*dy/dx-den*u_x[Nx-1,j])*phi_xL+(Gamma*dx/dy+den*u_y[Nx-1,j])*phi[Nx-1,j-1]+(Gamma*dx/dy)*phi[Nx-1,j+1])/(Gamma * (dy/dx + 2*dy/dx + dx/dy + dx/dy) + den*(u_y[Nx-1,j]))

            # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = ((2*Gamma*dy/dx+den*u_x[0, j])*phi_x0+ (Gamma*dy/dx)*phi[1,j]+(Gamma*dx/dy+den*u_y[0, j])*phi[0,j-1]+(Gamma*dx/dy)*phi[0,j+1])/(Gamma * (2*dy/dx + dy/dx + dx/dy + dx/dy) + den*(u_x[0, j]+u_y[0, j]))

            # North-eastern corner
                phi[Nx-1,Ny-1] = ((Gamma*dy/dx+den*u_x[Nx-1,Ny-1])*phi[Nx-1,Ny-1]+ (2*Gamma*dy/dx-den*u_x[Nx-1,Ny-1])*phi_xL+(Gamma*dx/dy+den*u_y[Nx-1,Ny-1])*phi[Nx-1,Ny-2]+(2*Gamma*dx/dy)*phi_yL-den*u_y[Nx-1,Ny-1])/(Gamma * (dy/dx + 2*dy/dx + 2*dx/dy + dx/dy))

            # South-eastern corner
                phi[Nx-1, 0] = ((Gamma*dy/dx+den*u_x[Nx-1, 0])*phi[Nx-2,0]+ (2*Gamma*dy/dx-den*u_x[Nx-1, 0])*phi_xL+(2*Gamma*dx/dy+den*u_y[Nx-1, 0])*phi_y0+(Gamma*dx/dy)*phi[Nx-1,1])/(Gamma * (dy/dx + 2*dy/dx + 2*dx/dy + dx/dy) + den*(u_x[Nx-1, 0]))

            # North-western corner
                phi[0, Ny - 1] = ((2*Gamma*dy/dx+den*u_x[0, Ny - 1])*phi_x0+ (Gamma*dy/dx)*phi[1,Ny-1]+(Gamma*dx/dy+den*u_y[0, Ny - 1])*phi[0,Ny-2]+(2*Gamma*dx/dy)*phi_yL-den*u_y[0, Ny - 1])/(Gamma * (2*dy/dx + dy/dx + 2*dx/dy + dx/dy) + den*(u_x[0, Ny - 1]))

            # South-western corner
                phi[0, 0] = ((2*Gamma*dy/dx+den*u_x[0, 0])*phi_x0+ (Gamma*dy/dx)*phi[1,0]+(2*Gamma*dx/dy+den*u_y[0, 0])*phi_y0+(Gamma*dx/dy)*phi[0,1])/(Gamma * (2*dy/dx + dy/dx + 2*dx/dy + dx/dy) + den*(u_x[0, 0]+u_y[0, 0]))

                sum=0
                for i in range(0,Nx):
                    for j in range(0,Ny):
                        sum += (phi[i,j]-old_phi[i,j])**2
                er = np.sqrt(sum/(Nx*Ny))

        # Case where Gamma = 0
        else:
            for iteration in range(max_iterations):
                old_phi = phi.copy()
                # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = (u_x[i, j]*phi[i-1,j] + u_y[i, j]*phi[i,j-1])/(u_x[i, j]+u_y[i, j])

                # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = (u_x[i, j]*phi[i-1,0] + u_y[i, j]*phi_y0)/(u_x[i, j]+u_y[i, j])

                # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny - 1] = (u_x[i, Ny - 1]*phi[i-1,Ny-1] + u_y[i, Ny - 1]*phi[i,Ny-2] - u_y[i, Ny - 1]*phi_yL)/u_x[i, Ny - 1]

                # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx - 1, j] = (u_x[Nx - 1, j]*phi[Nx-2,j] + u_y[Nx - 1, j]*phi[Nx-1,j-1] - u_x[Nx - 1, j]*phi_xL)/u_y[Nx - 1, j]

                # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = (u_x[0, j]*phi_x0 + u_y[0, j]*phi[0,j-1])/(u_x[0, j]+u_y[0, j])

                # North-eastern corner
                phi[Nx - 1, Ny - 1] = (u_x[Nx - 1, Ny - 1]*phi_xL+u_y[Nx - 1, Ny - 1]*phi_yL)/(u_x[Nx - 1, Ny - 1]+u_y[Nx - 1, Ny - 1])

                # South-eastern corner
                phi[Nx - 1, 0] = (u_x[Nx - 1, 0] * phi[Nx-2,0] + u_y[Nx - 1, 0] * phi_y0 - u_x[Nx - 1, 0] * phi_xL)/u_y[Nx - 1, 0]

                # North-western corner
                phi[0, Ny - 1] = (u_x[Nx - 1, 0] * phi_x0 + u_y[Nx - 1, 0]*phi[0,Ny-2] - u_y[Nx - 1, 0]*phi_yL)/u_x[Nx - 1, 0]

                # South-western corner
                phi[0, 0] = (u_x[0, 0]*phi_x0 + u_y[0, 0]*phi_y0)/(u_x[0, 0]+u_y[0, 0])

                sum = 0
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sum += (phi[i, j] - old_phi[i, j]) ** 2
                er = np.sqrt(sum / (Nx * Ny))

                # Convergence check
                er = np.sqrt(np.mean((phi - old_phi) ** 2))
                if er < tolerance:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

            # Central Difference Scheme


    else:
        # Case when gamma is not zero
        if Gamma != 0:
            for iteration in range(max_iterations):
                old_phi = phi.copy()

                # Debug: Print phi at the start of each iteration
                #print(f"Iteration {iteration} - Start: phi =", phi)

                # Inner elements
                for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):
                        phi[i, j] = ((Gamma * dy / dx + 0.5 * den * u_x[i, j]) * phi[i - 1, j] + (Gamma * dy / dx - 0.5 * den * u_x[i, j]) * phi[i + 1, j] + (
                                    Gamma * dx / dy + 0.5 * den * u_y[i, j]) * phi[i, j - 1] + (Gamma * dx / dy - 0.5*den*u_y[i, j]) * phi[i, j + 1]) / (
                                                Gamma * (dy / dx + dy / dx + dx / dy + dx / dy) )
                # Debug: Print phi after updating inner elements
                #print(f"Iteration {iteration} - After inner update: phi =", phi)

                # Southern boundary
                for i in range(1, Nx - 1):
                    phi[i, 0] = ((Gamma * dy / dx + 0.5 * den * u_x[i, 0]) * phi[i - 1, 0] + (Gamma * dy / dx - 0.5 * den * u_x[i, 0]) * phi[i + 1, 0] + (
                                    2*Gamma * dx / dy +  den * u_y[i, 0]) * phi_y0 + (Gamma * dx / dy - 0.5*den*u_y[i, 0]) * phi[i, 1]) / (
                                                Gamma * (dy / dx + dy / dx + dx / dy + 2*dx / dy) - 0.5 * den * u_y[i, 0])

                # Northern boundary
                for i in range(1, Nx - 1):
                    phi[i, Ny - 1] = ((Gamma * dy / dx + 0.5 * den * u_x[i, Ny - 1]) * phi[i - 1, Ny-1] + (Gamma * dy / dx - 0.5 * den * u_x[i, Ny - 1]) * phi[i + 1, Ny-1] + (
                                    Gamma * dx / dy + 0.5 * den * u_y[i, Ny - 1]) * phi[i, Ny-2] + (2*Gamma * dx / dy - den*u_y[i, Ny - 1]) * phi_yL) / (
                                                Gamma * (dy / dx + dy / dx + 2*dx / dy + dx / dy) - 0.5 * den * u_y[i, Ny - 1] )

                # Eastern boundary
                for j in range(1, Ny - 1):
                    phi[Nx - 1, j] = ((Gamma * dy / dx + 0.5 * den * u_x[Nx - 1, j]) * phi[Nx-2, j] + (2*Gamma * dy / dx - den * u_x[Nx - 1, j]) * phi_xL + (
                                    Gamma * dx / dy + 0.5 * den * u_y[Nx - 1, j]) * phi[Nx-1, j - 1] + (Gamma * dx / dy - 0.5*den*u_y[Nx - 1, j]) * phi[Nx-1, j + 1]) / (
                                                Gamma * (dy / dx + 2*dy / dx + dx / dy + dx / dy) - 0.5 * den * u_x[Nx - 1, j] )

                # Western boundary
                for j in range(1, Ny - 1):
                    phi[0, j] = ((2* Gamma*dy / dx + den * u_x[0, j]) * phi_x0 + (Gamma * dy / dx - 0.5 * den * u_x[0, j]) * phi[1, j] + (
                                    Gamma * dx / dy + 0.5 * den * u_y[0, j]) * phi[0, j - 1] + (Gamma * dx / dy - 0.5*den*u_y[0, j]) * phi[0, j + 1]) / (
                                                Gamma * (2*dy / dx + dy / dx + dx / dy + dx / dy) -0.5*den*u_x[0, j] )
                # Debug: Print phi after updating inner elements
               # print(f"Iteration {iteration} - After inner update: phi =", phi)

                # North-eastern corner
                phi[Nx - 1, Ny - 1] = ((Gamma * dy / dx + 0.5 * den * u_x[Nx - 1, Ny - 1]) * phi[Nx-2, Ny-1] + (2*Gamma * dy / dx - den * u_x[Nx - 1, Ny - 1]) * phi_xL + (
                                    Gamma * dx / dy + 0.5 * den * u_y[Nx - 1, Ny - 1]) * phi[Nx-1, Ny-2] + (2*Gamma * dx / dy - den*u_y[Nx - 1, Ny - 1]) * phi_yL) / (
                                                Gamma * (2*dy / dx + dy / dx + 2*dx / dy + dx / dy) - 0.5 * den * (u_x[Nx - 1, Ny - 1]+u_y[Nx - 1, Ny - 1]) )

                # South-eastern corner
                phi[Nx - 1, 0] = ((Gamma * dy / dx + 0.5 * den * u_x[Nx - 1, 0]) * phi[Nx-2, 0] + (2*Gamma * dy / dx - den * u_x[Nx - 1, 0]) * phi_xL + (
                                    2*Gamma * dx / dy + den * u_y[Nx - 1, 0]) * phi_y0 + (Gamma * dx / dy - 0.5*den*u_y[Nx - 1, 0]) * phi[Nx-1, 1]) / (
                                                Gamma * (dy / dx + 2*dy / dx + 2*dx / dy + dx / dy) - 0.5 * den * (u_x[Nx - 1, 0]+u_y[Nx - 1, 0]) )

                # North-western corner
                phi[0, Ny - 1] = ((2 * Gamma * dy / dx + den * u_x[0, Ny - 1]) * phi_x0 + (Gamma * dy / dx) * phi[1, Ny - 1] + (
                            Gamma * dx / dy + den * u_y[0, Ny - 1]) * phi[0, Ny - 2] + (2 * Gamma * dx / dy) * phi_yL - den * u_y[0, Ny - 1]) / (
                                             Gamma * (2 * dy / dx + dy / dx + 2 * dx / dy + dx / dy) + den * (u_x[0, Ny - 1]+u_y[0, Ny - 1]))

                # South-western corner
                phi[0, 0] = ((2* Gamma*dy / dx + den * u_x[0, 0]) * phi_x0 + (Gamma * dy / dx - 0.5 * den * u_x[0, 0]) * phi[1, 0] + (
                                    2*Gamma * dx / dy + den * u_y[0, 0]) * phi_y0 + (Gamma * dx / dy - 0.5*den*u_y[0, 0]) * phi[0, 1]) / (
                                                Gamma * (2*dy / dx + dy / dx + 2*dx / dy + dx / dy) -0.5*den*(u_x[0, 0]+u_y[0, 0]) )

                sum = 0
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        sum += (phi[i, j] - old_phi[i, j]) ** 2
                er = np.sqrt(sum / (Nx * Ny))

                #print(f"Iteration # {iteration} and max error is {er}")
                #print(f"NW= {phi[0, Ny - 1]} NE={phi[Nx - 1, Ny - 1]} and max error is {er}")
                # Insert the check here, after the main computation steps
                if np.isnan(phi).any() or np.isinf(phi).any() or np.max(np.abs(phi)) > 1e10:
                    print("Warning: NaN, Inf, or extreme value found in phi")
                    break  # Optionally break the loop to stop further computation
                # Convergence check
                er = np.sqrt(np.mean((phi - old_phi) ** 2))
                if er < tolerance:
                    print(f"Converged after {iteration + 1} iterations.")
                    break
        # Case where Gamma = 0
        else:
            print("CDS not possible for Gamma=0")

        # Convert to CSR format for efficient arithmetic operations
        #A_csr = A.tocsr()
        #print("A matrix:")
        #print(A)
        #print("A_csr matrix:")
        #print(A_csr)
        #print("\nb vector:")
        #print(b)
        # Solve the system using bicgstab
        #phi_flat, _ = bicgstab(A_csr, b)
        #phi = phi_flat.reshape((Nx, Ny))

    return phi, u_x, u_y

# Plotting
# Use func and get u_x, u_y for plotting
Nx = 80
Ny = 80
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
phi80, u_x, u_y = func(Nx, Ny)

# Now use u_x and u_y for plotting
plt.contourf(x, y, np.sqrt(u_x ** 2 + u_y ** 2), cmap='jet', levels=50)
plt.colorbar(label='Velocity Magnitude')
plt.title('Rotational Velocity Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Define your grids
x80 = np.linspace(0, Lx, 80)
y80 = np.linspace(0, Ly, 80)
x160 = np.linspace(0, Lx, 160)
y160 = np.linspace(0, Ly, 160)
x320 = np.linspace(0, Lx, 320)
y320 = np.linspace(0, Ly, 320)

# Generate data
phi80 = func(80, 80)[0]   # Only take the first element (phi) from the returned tuple
phi160 = func(160, 160)[0] # Same here
phi320 = func(320, 320)[0] # And here

# Plot phi80
plt.figure(figsize=(6, 6))
plt.contourf(phi80, cmap='jet', levels=50)
plt.colorbar(label='Phi Value')
plt.title('Phi Distribution for 80x80 Grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plot phi160
plt.figure(figsize=(6, 6))
plt.contourf(phi160, cmap='jet', levels=50)
plt.colorbar(label='Phi Value')
plt.title('Phi Distribution for 160x160 Grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plot phi320
plt.figure(figsize=(6, 6))
plt.contourf(phi320, cmap='jet', levels=50)
plt.colorbar(label='Phi Value')
plt.title('Phi Distribution for 320x320 Grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Create interpolators
f80 = RGI((x80, y80), phi80)
f160 = RGI((x160, y160), phi160)

# Perform interpolation
phi80_interp = f80(np.array(np.meshgrid(x320, y320)).T.reshape(-1, 2)).reshape(320, 320)
phi160_interp = f160(np.array(np.meshgrid(x320, y320)).T.reshape(-1, 2)).reshape(320, 320)

# After calculating phi80_interp and phi160_interp
print("Sample of phi320:", phi320[::40, ::40])  # Print a sample of every 40th element
print("Sample of phi80_interp:", phi80_interp[::40, ::40])
print("Sample of phi160_interp:", phi160_interp[::40, ::40])


# Error calculations
ec = np.sqrt(np.mean((phi320 - phi80_interp)**2))
ef = np.sqrt(np.mean((phi320 - phi160_interp)**2))

print("current error for 80 interpolation:", ec)
print("current error for 160 interpolation:", ef)
# Error calculations using maximum difference
ec_max = np.max(abs(phi320 - phi80_interp))
ef_max = np.max(abs(phi320 - phi160_interp))

print("Maximum error for 80 interpolation:", ec_max)
print("Maximum error for 160 interpolation:", ef_max)

# Adding a small constant to avoid division by zero
epsilon = 1e-10

# Check if ec or ef is very close to zero and handle accordingly
if abs(ec) < epsilon or abs(ef) < epsilon:
    print("Error calculation resulted in a very small value, unable to compute order of convergence.")
else:
    # Order of convergence
    O = np.log(ec / ef) / np.log((Lx / 80) / (Lx / 160))
    print(f"Order of convergence is {O}")
