import jax
import jax.numpy as jnp
import jax.scipy.ndimage
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Ensure JAX uses Float64 for physical precision matching your Julia implementation
jax.config.update("jax_enable_x64", True)

class TaylorGreenVortex3D:
    def __init__(self, N=32, Re=1600.0, L=1.0, V0=1.0):
        self.N = N
        self.Re = Re
        self.L = L
        self.V0 = V0
        self.nu = (V0 * L) / Re
        
        # Domain configuration [0, 2*pi*L]
        self.domain_size = 2.0 * jnp.pi * L
        self.dx = self.domain_size / N
        self.dy = self.domain_size / N
        self.dz = self.domain_size / N
        
        # Grid setup (collocated structural representation)
        self.x = jnp.linspace(0, self.domain_size, N, endpoint=False)
        self.y = jnp.linspace(0, self.domain_size, N, endpoint=False)
        self.z = jnp.linspace(0, self.domain_size, N, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Precompute structural FFT wavenumbers for the exact Pressure Poisson Step
        kx = 2.0 * jnp.sin(jnp.fft.fftfreq(N) * jnp.pi) / self.dx
        ky = 2.0 * jnp.sin(jnp.fft.fftfreq(N) * jnp.pi) / self.dy
        kz = 2.0 * jnp.sin(jnp.fft.fftfreq(N) * jnp.pi) / self.dz
        KX, KY, KZ = jnp.meshgrid(kx**2, ky**2, kz**2, indexing='ij')
        self.k_squared = KX + KY + KZ
        self.k_squared = self.k_squared.at[0, 0, 0].set(1.0) # Avoid division by zero

    def get_initial_conditions(self):
        u = self.V0 * jnp.sin(self.X / self.L) * jnp.cos(self.Y / self.L) * jnp.cos(self.Z / self.L)
        v = -self.V0 * jnp.cos(self.X / self.L) * jnp.sin(self.Y / self.L) * jnp.cos(self.Z / self.L)
        w = jnp.zeros_like(u)
        return u, v, w

    @staticmethod
    def advect_semi_lagrangian(field, u, v, w, dt, dx, dy, dz):
        Nx, Ny, Nz = field.shape
        idx_x, idx_y, idx_z = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz), indexing='ij')
        
        # Backtrack coordinates in index space
        coords_x = idx_x - u * (dt / dx)
        coords_y = idx_y - v * (dt / dy)
        coords_z = idx_z - w * (dt / dz)
        
        coords = jnp.stack([coords_x, coords_y, coords_z], axis=0)
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    @staticmethod
    def laplacian_3d(f, dx, dy, dz):
        d2f_dx2 = (jnp.roll(f, -1, axis=0) - 2.0 * f + jnp.roll(f, 1, axis=0)) / (dx**2)
        d2f_dy2 = (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / (dy**2)
        d2f_dz2 = (jnp.roll(f, -1, axis=2) - 2.0 * f + jnp.roll(f, 1, axis=2)) / (dy**2)
        return d2f_dx2 + d2f_dy2 + d2f_dz2

    @staticmethod
    def central_grads_3d(f, dx, dy, dz):
        df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
        df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
        df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dz)
        return df_dx, df_dy, df_dz

    def project_pressure_fft(self, u_star, v_star, w_star, dt):
        du_dx = (jnp.roll(u_star, -1, axis=0) - jnp.roll(u_star, 1, axis=0)) / (2.0 * self.dx)
        dv_dy = (jnp.roll(v_star, -1, axis=1) - jnp.roll(v_star, 1, axis=1)) / (2.0 * self.dy)
        dw_dz = (jnp.roll(w_star, -1, axis=2) - jnp.roll(w_star, 1, axis=2)) / (2.0 * self.dz)
        div = du_dx + dv_dy + dw_dz
        
        div_hat = jnp.fft.fftn(div)
        p_hat = -div_hat / self.k_squared
        p_hat = p_hat.at[0, 0, 0].set(0.0)
        
        p = jnp.real(jnp.fft.ifftn(p_hat))
        return p

    def compute_diagnostics(self, u, v, w):
        ke = 0.5 * (u**2 + v**2 + w**2)
        E_k = jnp.mean(ke)
        
        _, du_dy, du_dz = self.central_grads_3d(u, self.dx, self.dy, self.dz)
        dv_dx, _, dv_dz = self.central_grads_3d(v, self.dx, self.dy, self.dz)
        dw_dx, dw_dy, _ = self.central_grads_3d(w, self.dx, self.dy, self.dz)
        
        rot_x = dw_dy - dv_dz
        rot_y = du_dz - dw_dx
        rot_z = dv_dx - du_dy
        
        enstrophy = jnp.mean(rot_x**2 + rot_y**2 + rot_z**2)
        epsilon = 2.0 * self.nu * enstrophy
        return E_k, epsilon

    def step(self, state, _):
        u, v, w, dt = state
        
        # 1. Semi-Lagrangian Advection
        u_adv = self.advect_semi_lagrangian(u, u, v, w, dt, self.dx, self.dy, self.dz)
        v_adv = self.advect_semi_lagrangian(v, u, v, w, dt, self.dx, self.dy, self.dz)
        w_adv = self.advect_semi_lagrangian(w, u, v, w, dt, self.dx, self.dy, self.dz)
        
        # 2. Viscous Diffusion Step
        u_star = u_adv + dt * self.nu * self.laplacian_3d(u_adv, self.dx, self.dy, self.dz)
        v_star = v_adv + dt * self.nu * self.laplacian_3d(v_adv, self.dx, self.dy, self.dz)
        w_star = w_adv + dt * self.nu * self.laplacian_3d(w_adv, self.dx, self.dy, self.dz)
        
        # 3. FFT Pressure Projection
        p = self.project_pressure_fft(u_star, v_star, w_star, dt)
        dp_dx, dp_dy, dp_dz = self.central_grads_3d(p, self.dx, self.dy, self.dz)
        
        # 4. Velocity Correction Update
        u_next = u_star - dt * dp_dx
        v_next = v_star - dt * dp_dy
        w_next = w_star - dt * dp_dz
        
        E_k, epsilon = self.compute_diagnostics(u_next, v_next, w_next)
        return (u_next, v_next, w_next, dt), (E_k, epsilon)


# --- 2D Validation Plotting ---
def generate_validation_plot(time_axis, simulated_eps):
    ref_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ref_eps = np.array([0.00094, 0.0013, 0.0021, 0.0034, 0.0051, 0.0071, 0.0094, 0.0113, 0.0118, 0.0104, 0.0084])

    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, simulated_eps, label="JAX FDM Solver (32³)", color="#1f77b4", linewidth=2.5)
    plt.scatter(ref_t, ref_eps, color="#d62728", marker="o", s=50, zorder=5, label="Spectral DNS Reference")
    plt.plot(ref_t, ref_eps, color="#d62728", linestyle="--", alpha=0.6)

    plt.title("Taylor-Green Vortex Dissipation Rate $\epsilon(t)$ at $Re=1600$", fontsize=12, fontweight="bold")
    plt.xlabel("Dimensionless Time $t$", fontsize=11)
    plt.ylabel("Dissipation Rate $\epsilon$", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/tgv_dissipation.png", dpi=300, bbox_inches="tight")
    plt.close()


# --- Dual 3D Volumetric Visualizer (Hybrid Engine) ---
def generate_3d_structure_plot(sim, u, v, w):
    """Generates a true 3D fluid volume visualization.
    1. Saves an interactive HTML via Plotly (no system dependencies needed).
    2. Saves a beautiful cloud-like PNG via Matplotlib (bypasses Kaleido/Chrome completely).
    """
    print("Generating volumetric 3D structures...")
    
    # Convert JAX arrays to NumPy arrays
    u_np, v_np, w_np = np.array(u), np.array(v), np.array(w)
    v_mag = np.sqrt(u_np**2 + v_np**2 + w_np**2)
    
    # Flatten arrays for 3D coordinate mapping
    X_flat = np.array(sim.X).flatten()
    Y_flat = np.array(sim.Y).flatten()
    Z_flat = np.array(sim.Z).flatten()
    values_flat = v_mag.flatten()
    
    # Calculate thresholds to strip out slow-moving fluid
    val_min = v_mag.min()
    val_max = v_mag.max()
    isomin_threshold = val_min + 0.15 * (val_max - val_min) 

    # --- 1. OUTPUT INTERACTIVE HTML (via Plotly) ---
    fig_plotly = go.Figure(data=go.Volume(
        x=X_flat, y=Y_flat, z=Z_flat, value=values_flat,
        isomin=float(isomin_threshold), isomax=float(val_max),
        opacity=0.2, surface_count=25, colorscale='Turbo',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig_plotly.update_layout(
        title=f"3D Fluid Structure (Velocity Magnitude)<br>Resolution: {sim.N}³, Re={sim.Re}",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    os.makedirs("assets", exist_ok=True)
    fig_plotly.write_html("assets/tgv_3d_structure.html")
    print("Interactive 3D volumetric plot written to: assets/tgv_3d_structure.html")
    
    # --- 2. OUTPUT STATIC PNG (via Native Matplotlib 3D Alpha-Cloud) ---
    # We construct a dense semi-transparent scatter cloud that looks exactly like a 3D raytraced volume.
    fig_mpl = plt.figure(figsize=(10, 8))
    ax = fig_mpl.add_subplot(111, projection='3d')
    
    # Filter points to only show structures above the threshold
    mask = values_flat > isomin_threshold
    
    # Scatter plot with large, overlapping, alpha-blended markers creates a gas/volume effect
    sc = ax.scatter(
        X_flat[mask], Y_flat[mask], Z_flat[mask],
        c=values_flat[mask],
        cmap='turbo',          # High-contrast colormap mimicking the reference target
        alpha=0.15,            # Low alpha values stack together visually to form density
        s=35,                  # Point size scaled to slightly overlap at 32^3
        edgecolors='none'
    )
    
    ax.set_xlim(0, sim.domain_size)
    ax.set_ylim(0, sim.domain_size)
    ax.set_zlim(0, sim.domain_size)
    
    ax.set_xlabel('X Coordinate', fontweight='bold', labelpad=10)
    ax.set_ylabel('Y Coordinate', fontweight='bold', labelpad=10)
    ax.set_zlabel('Z Coordinate', fontweight='bold', labelpad=10)
    ax.set_title(f"3D Fluid Structure Volume (Velocity Magnitude)\nResolution: {sim.N}³, Re={sim.Re}", 
                 fontsize=12, fontweight='bold', pad=10)
    
    cbar = fig_mpl.colorbar(sc, ax=ax, shrink=0.55, aspect=14, pad=0.1)
    cbar.set_label('Velocity Magnitude $|\\mathbf{u}|$', fontsize=10)
    
    ax.view_init(elev=25, azim=-45)
    
    png_path = "assets/tgv_3d_structure_static.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Static 3D volumetric image successfully written to: {png_path} (Bypassed Kaleido!)")


# --- Main Driver ---
if __name__ == "__main__":
    N_resolution = 32
    sim = TaylorGreenVortex3D(N=N_resolution, Re=1600.0)
    u0, v0, w0 = sim.get_initial_conditions()
    
    dt_adv = 0.1 * (sim.dx / sim.V0)
    dt_visc = 0.25 * (sim.dx**2 / (6.0 * sim.nu))
    dt = float(min(dt_adv, dt_visc))
    
    t_max = 10.0
    steps = int(t_max / dt)
    time_axis = np.linspace(0, t_max, steps)
    
    print(f"Compiling transient solver loop...")
    run_sim = jax.jit(lambda u, v, w: jax.lax.scan(sim.step, (u, v, w, dt), None, length=steps))
    
    # Run simulation execution
    _, _ = run_sim(u0, v0, w0) # Compilation warm up
    print("Simulating TGV progression across time domains...")
    (u_f, v_f, w_f, _), (E_history, eps_history) = run_sim(u0, v0, w0)
    jax.block_until_ready((u_f, v_f, w_f))
    
    print("Generating graphics artifacts...")
    generate_validation_plot(time_axis, np.array(eps_history))
    generate_3d_structure_plot(sim, u_f, v_f, w_f)
    print("Done!")