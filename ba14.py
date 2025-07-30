# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 08:30:16 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Enhanced Publication Quality 3D Battery Pack Thermal Analysis
============================================================
- Separated plots for better clarity
- Enhanced boundary visualization with color coding
- Improved aesthetics for all figures
- No text labels on battery cells for cleaner look

Created for publication use
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from fipy import Grid3D, CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
from fipy.solvers import LinearGMRESSolver, LinearPCGSolver
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================

# Enhanced figure settings for publication
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15
})

# Create output directory
OUTPUT_DIR = 'battery_analysis_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced color schemes
COLORS = {
    'battery': '#FFD700',      # Gold
    'air': '#E6F3FF',         # Light blue
    'convection': '#FF4500',   # Orange red
    'insulated': '#87CEEB',    # Sky blue
    'heat': '#DC143C',         # Crimson
    'max_temp': '#8B0000',     # Dark red
    'avg_temp': '#FF8C00',     # Dark orange
    'air_temp': '#4682B4'      # Steel blue
}

# ============================================================================
# GLOBAL PARAMETERS (unchanged)
# ============================================================================

# Geometry Parameters (meters)
CELL_WIDTH = 0.065          # Individual cell width (65 mm)
CELL_HEIGHT = 0.095         # Individual cell height (95 mm)  
CELL_THICKNESS = 0.027      # Individual cell thickness (27 mm)
GAP_BETWEEN_CELLS = 0.0325  # Air gap between cells (32.5 mm)
NX_CELLS = 3                # Number of cells in x-direction
NY_CELLS = 2                # Number of cells in y-direction

# Mesh Parameters (optimized for publication quality)
MESH_NX = 104               # X-direction mesh points
MESH_NY = 46                # Y-direction mesh points
MESH_NZ = 38                # Z-direction mesh points

# Material Properties
K_CELL = 2.0                # Thermal conductivity of battery (W/m·K)
K_AIR = 0.026               # Thermal conductivity of air (W/m·K)
RHO_CELL = 2500.0           # Density of battery (kg/m³)
RHO_AIR = 1.2               # Density of air (kg/m³)
CP_CELL = 1000.0            # Specific heat of battery (J/kg·K)
CP_AIR = 1005.0             # Specific heat of air (J/kg·K)

# Operating Conditions
DISCHARGE_CURRENT = 30.0    # Discharge current (A)
INTERNAL_RESISTANCE = 0.012 # Internal resistance (Ω)
T_AMBIENT = 300.15           # Ambient temperature (K) = 27°C
T_INITIAL = 300.15           # Initial temperature (K) = 27°C

# Heat Transfer Coefficients for publication
H_NATURAL = 10.0            # Natural convection (W/m²·K)
H_FORCED = 50.0             # Forced air cooling (W/m²·K)
H_LIQUID = 250            # Liquid cooling (W/m²·K)

# Simulation Parameters
TIME_STEPS_DEFAULT = 1800  # Time steps for complete analysis
DT_DEFAULT = 2            # Time step size (seconds)
SOLVER_TOLERANCE = 1e-5     # Solver tolerance
SOLVER_MAX_ITERATIONS = 500

# Multiprocessing Parameters
N_PROCESSES = mp.cpu_count()

# ============================================================================
# PARALLEL PROCESSING UTILITIES (unchanged)
# ============================================================================

def parallel_region_identification(args):
    """Parallel function for identifying battery regions with smooth transitions."""
    i, j, mesh_info = args
    
    x_coords, y_coords, z_coords = mesh_info['coords']
    transition_width = mesh_info['transition_width']
    
    x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS)
    x_end = x_start + CELL_WIDTH
    y_start = j * (CELL_THICKNESS + GAP_BETWEEN_CELLS)
    y_end = y_start + CELL_THICKNESS
    
    x_trans_start = 0.5 * (1 + np.tanh((x_coords - x_start) / transition_width))
    x_trans_end = 0.5 * (1 - np.tanh((x_coords - x_end) / transition_width))
    y_trans_start = 0.5 * (1 + np.tanh((y_coords - y_start) / transition_width))
    y_trans_end = 0.5 * (1 - np.tanh((y_coords - y_end) / transition_width))
    
    battery_indicator = x_trans_start * x_trans_end * y_trans_start * y_trans_end
    
    return battery_indicator

def parallel_heat_calculation(args):
    """Parallel function for heat source calculation."""
    region_data, heat_density = args
    return heat_density * region_data

# ============================================================================
# ENHANCED PUBLICATION QUALITY BATTERY PACK CLASS
# ============================================================================

class PublicationBatteryPack3D:
    """Enhanced battery pack class with publication-quality visualization."""
    
    def __init__(self, nx_cells=NX_CELLS, ny_cells=NY_CELLS, use_multiprocessing=True):
        """Initialize battery pack for publication analysis."""
        self.nx_cells = nx_cells
        self.ny_cells = ny_cells
        self.use_multiprocessing = use_multiprocessing
        
        # Calculate pack dimensions
        self.pack_width = nx_cells * CELL_WIDTH + (nx_cells - 1) * GAP_BETWEEN_CELLS
        self.pack_depth = ny_cells * CELL_THICKNESS + (ny_cells - 1) * GAP_BETWEEN_CELLS
        self.pack_height = CELL_HEIGHT
        
        # Create uniform 3D mesh
        self.mesh = Grid3D(
            nx=MESH_NX, ny=MESH_NY, nz=MESH_NZ,
            dx=self.pack_width / MESH_NX,
            dy=self.pack_depth / MESH_NY,
            dz=self.pack_height / MESH_NZ
        )
        
        print(f"\n{'='*60}")
        print(f"ENHANCED PUBLICATION QUALITY 3D BATTERY ANALYSIS")
        print(f"{'='*60}")
        print(f"Pack: {nx_cells}×{ny_cells} cells, Mesh: {MESH_NX}×{MESH_NY}×{MESH_NZ}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        # Initialize
        self._identify_regions_parallel()
        self._create_material_fields_parallel()
        
    def _identify_regions_parallel(self):
        """Identify battery regions using parallel processing."""
        print(f"Identifying regions...")
        start_time = time.time()
        
        x, y, z = self.mesh.cellCenters
        mesh_info = {'coords': (x, y, z), 'transition_width': 0.001}
        
        self.battery_fraction = np.zeros(self.mesh.numberOfCells)
        
        if self.use_multiprocessing:
            args_list = [(i, j, mesh_info) for i in range(self.nx_cells) for j in range(self.ny_cells)]
            with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
                results = list(executor.map(parallel_region_identification, args_list))
            for result in results:
                self.battery_fraction += result
        else:
            for i in range(self.nx_cells):
                for j in range(self.ny_cells):
                    result = parallel_region_identification((i, j, mesh_info))
                    self.battery_fraction += result
        
        self.battery_fraction = np.clip(self.battery_fraction, 0, 1)
        self.is_battery = CellVariable(mesh=self.mesh, value=self.battery_fraction)
        self.battery_mask = self.battery_fraction > 0.5
        
        elapsed = time.time() - start_time
        n_battery = np.sum(self.battery_mask)
        print(f"  Completed in {elapsed:.2f}s, Battery cells: {n_battery:,}")
        
    def _create_material_fields_parallel(self):
        """Create material property fields using parallel processing."""
        print(f"Creating material fields...")
        start_time = time.time()
        
        bf = self.battery_fraction
        
        if self.use_multiprocessing:
            n_cells = len(bf)
            chunk_size = max(1, n_cells // N_PROCESSES)
            
            with ThreadPoolExecutor(max_workers=N_PROCESSES) as executor:
                def process_k_chunk(chunk_data):
                    start_idx, end_idx = chunk_data
                    bf_chunk = bf[start_idx:end_idx]
                    k_chunk = np.zeros_like(bf_chunk)
                    
                    k_chunk[bf_chunk > 0.99] = K_CELL
                    k_chunk[bf_chunk < 0.01] = K_AIR
                    interface_mask = (bf_chunk >= 0.01) & (bf_chunk <= 0.99)
                    if np.any(interface_mask):
                        k_harmonic = 2 * K_CELL * K_AIR / (K_CELL + K_AIR)
                        k_chunk[interface_mask] = K_AIR + bf_chunk[interface_mask] * (k_harmonic - K_AIR)
                    
                    return k_chunk
                
                chunks = [(i, min(i + chunk_size, n_cells)) for i in range(0, n_cells, chunk_size)]
                k_results = list(executor.map(process_k_chunk, chunks))
                k_values = np.concatenate(k_results)
        else:
            k_values = np.zeros_like(bf)
            k_values[bf > 0.99] = K_CELL
            k_values[bf < 0.01] = K_AIR
            interface_mask = (bf >= 0.01) & (bf <= 0.99)
            if np.any(interface_mask):
                k_harmonic = 2 * K_CELL * K_AIR / (K_CELL + K_AIR)
                k_values[interface_mask] = K_AIR + bf[interface_mask] * (k_harmonic - K_AIR)
        
        self.k = CellVariable(mesh=self.mesh, value=k_values)
        self.rho = CellVariable(mesh=self.mesh, value=RHO_AIR + bf * (RHO_CELL - RHO_AIR))
        self.cp = CellVariable(mesh=self.mesh, value=CP_AIR + bf * (CP_CELL - CP_AIR))
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s")
        
    def build_heat_source_parallel(self):
        """Build heat generation field using parallel processing."""
        print(f"Building heat source...")
        start_time = time.time()
        
        cell_volume = CELL_WIDTH * CELL_HEIGHT * CELL_THICKNESS
        heat_per_cell = DISCHARGE_CURRENT**2 * INTERNAL_RESISTANCE
        heat_density = heat_per_cell / cell_volume
        
        if self.use_multiprocessing:
            n_cells = len(self.battery_fraction)
            chunk_size = max(1, n_cells // N_PROCESSES)
            
            with ThreadPoolExecutor(max_workers=N_PROCESSES) as executor:
                chunks = [(self.battery_fraction[i:i+chunk_size], heat_density) 
                         for i in range(0, n_cells, chunk_size)]
                Q_results = list(executor.map(parallel_heat_calculation, chunks))
                Q_values = np.concatenate(Q_results)
        else:
            Q_values = heat_density * self.battery_fraction
        
        self.Q = CellVariable(mesh=self.mesh, value=Q_values)
        
        # Apply smoothing
        Q_3d = self.Q.value.reshape(MESH_NX, MESH_NY, MESH_NZ)
        Q_smooth = gaussian_filter(Q_3d, sigma=0.5)
        self.Q.setValue(Q_smooth.flatten())
        
        elapsed = time.time() - start_time
        total_heat = numerix.sum(self.Q.value * self.mesh.cellVolumes)
        expected_heat = heat_per_cell * self.nx_cells * self.ny_cells
        
        print(f"  Completed in {elapsed:.2f}s, Total heat: {total_heat:.2f} W")
        
    def solve_temperature(self, cooling_type='natural', time_steps=TIME_STEPS_DEFAULT, dt=DT_DEFAULT):
        """Solve temperature with correct boundary conditions."""
        if not hasattr(self, 'Q'):
            self.build_heat_source_parallel()
            
        h_map = {'natural': H_NATURAL, 'forced': H_FORCED, 'liquid': H_LIQUID}
        h = h_map.get(cooling_type, H_NATURAL)
        
        print(f"\nSolving {cooling_type.upper()} cooling (h={h} W/m²K)...")
        
        # Initialize temperature
        self.T = CellVariable(mesh=self.mesh, value=T_INITIAL, hasOld=True)
        
        # Create boundary conditions
        x, y, z = self.mesh.cellCenters
        h_eff = CellVariable(mesh=self.mesh, value=0.0)
        
        nx_mesh, ny_mesh, nz_mesh = MESH_NX, MESH_NY, MESH_NZ
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        h_eff_array = np.zeros(self.mesh.numberOfCells)
        h_eff_3d = h_eff_array.reshape(nx_mesh, ny_mesh, nz_mesh, order='F')
        
        # Apply convection only on side walls
        h_eff_3d[0, :, :] += h / dx      # Left
        h_eff_3d[-1, :, :] += h / dx     # Right
        h_eff_3d[:, 0, :] += h / dy      # Front
        h_eff_3d[:, -1, :] += h / dy     # Back
        # Top and bottom remain insulated
        
        h_eff.setValue(h_eff_3d.flatten())
        self.boundary_mask = h_eff.value > 0
        self.h_eff = h_eff
        
        # Heat equation
        eq = TransientTerm(coeff=self.rho * self.cp) == \
             DiffusionTerm(coeff=self.k) + self.Q - \
             ImplicitSourceTerm(coeff=h_eff) + h_eff * T_AMBIENT
        
        solver = LinearGMRESSolver(tolerance=SOLVER_TOLERANCE, 
                                  iterations=SOLVER_MAX_ITERATIONS,
                                  precon='jacobi')
        
        # Time stepping
        self.time_history = []
        self.temp_history = []
        
        start_time = time.time()
        
        for step in range(time_steps):
            self.T.updateOld()
            
            try:
                res = eq.solve(var=self.T, dt=dt, solver=solver)
            except:
                try:
                    solver_pcg = LinearPCGSolver(tolerance=SOLVER_TOLERANCE*10)
                    eq.solve(var=self.T, dt=dt, solver=solver_pcg)
                except:
                    continue
            
            # Store results
            if step % max(1, time_steps // 20) == 0:
                self.time_history.append((step + 1) * dt)
                self.temp_history.append(self.T.value.copy())
                
                if step % max(1, time_steps // 10) == 0:
                    T_max = numerix.max(self.T.value) -273.15
                    T_avg_battery = numerix.mean(self.T.value[self.battery_mask]) - 273.15
                    progress = (step + 1) / time_steps * 100
                    print(f"  [{progress:5.1f}%] Tmax={T_max:6.2f}°C, Tbatt={T_avg_battery:6.2f}°C")
        
        elapsed = time.time() - start_time
        print(f"  Solved in {elapsed:.1f}s")

    # ============================================================================
    # ENHANCED VISUALIZATION METHODS (Separated Plots)
    # ============================================================================
        
    def plot_temperature_history_publication(self, cooling_type='natural'):
        """Plot enhanced publication-quality temperature history."""
        if not hasattr(self, 'time_history') or len(self.time_history) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(11, 8))
        
        time_points = self.time_history
        T_max_history = []
        T_battery_history = []
        T_air_history = []
        
        for T_field in self.temp_history:
            T_celsius = T_field - 273.15
            T_max_history.append(np.max(T_celsius))
            T_battery_history.append(np.mean(T_celsius[self.battery_mask]))
            T_air_history.append(np.mean(T_celsius[~self.battery_mask]))
        
        # Enhanced plotting with gradient effect
        ax.plot(time_points, T_max_history, color=COLORS['max_temp'], linewidth=3.5, 
                label='Maximum Temperature', marker='o', markersize=8, markevery=len(time_points)//10)
        ax.plot(time_points, T_battery_history, color=COLORS['avg_temp'], linewidth=3, 
                label='Battery Average', linestyle='--', marker='s', markersize=7, markevery=len(time_points)//10)
        ax.plot(time_points, T_air_history, color=COLORS['air_temp'], linewidth=2.5, 
                label='Air Average', linestyle='-.', marker='^', markersize=6, markevery=len(time_points)//10)
        
        # Fill areas for better visualization
        ax.fill_between(time_points, 27, T_max_history, alpha=0.15, color=COLORS['max_temp'])
        ax.fill_between(time_points, 27, T_battery_history, alpha=0.1, color=COLORS['avg_temp'])
        
        # Reference line with style
        ax.axhline(y=27, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Initial (27°C)')
        
        # Enhanced formatting
        ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
        ax.set_title(f'Temperature Evolution - {cooling_type.title()} Cooling', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best', fontsize=13)
        
        # Add background gradient
        ax.set_facecolor('#F5F5F5')
        
        # Save figure
        filename = f'{OUTPUT_DIR}/temp_history_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_temperature_contour_publication(self, cooling_type='natural'):
        """Plot enhanced temperature contour separately."""
        T_3d = self.T.value.reshape(MESH_NX, MESH_NY, MESH_NZ, order='F')
        T_celsius = T_3d - 273.15
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        z_mid = MESH_NZ // 2
        T_xy = T_celsius[:, :, z_mid]
        T_xy_smooth = gaussian_filter(T_xy, sigma=1.5)
        
        x_mm = np.linspace(0, self.pack_width*1000, MESH_NX)
        y_mm = np.linspace(0, self.pack_depth*1000, MESH_NY)
        X_mm, Y_mm = np.meshgrid(x_mm, y_mm, indexing='ij')
        
        # Enhanced contour plot
        levels = np.linspace(T_xy_smooth.min(), T_xy_smooth.max(), 25)
        im = ax.contourf(X_mm, Y_mm, T_xy_smooth, levels=levels, cmap='hot_r', extend='both')
        cs = ax.contour(X_mm, Y_mm, T_xy_smooth, levels=10, colors='black', linewidths=1, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=10, fmt='%.1f°C')
        
        # Enhanced battery rectangles
        self._add_enhanced_battery_rectangles(ax)
        
        # Formatting
        ax.set_xlabel('Width (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Depth (mm)', fontsize=16, fontweight='bold')
        ax.set_title(f'Temperature Distribution at Mid-Height - {cooling_type.title()} Cooling', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Temperature (°C)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/temp_contour_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_temperature_profile_publication(self, cooling_type='natural'):
        """Plot enhanced temperature profile separately."""
        T_3d = self.T.value.reshape(MESH_NX, MESH_NY, MESH_NZ, order='F')
        T_celsius = T_3d - 273.15
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        z_mid = MESH_NZ // 2
        T_xy = T_celsius[:, :, z_mid]
        y_idx = MESH_NY // 2
        x_coords = np.linspace(0, self.pack_width*1000, MESH_NX)
        T_profile_x = T_xy[:, y_idx]
        T_profile_smooth = gaussian_filter(T_profile_x, sigma=2)
        
        # Main temperature profile with gradient
        ax.plot(x_coords, T_profile_smooth, color=COLORS['heat'], linewidth=4, 
                label='Temperature Profile', zorder=10)
        
        # Fill under curve
        ax.fill_between(x_coords, 27, T_profile_smooth, alpha=0.3, color=COLORS['heat'])
        
        # Enhanced battery region marking with gradient
        for i in range(self.nx_cells):
            x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS) * 1000
            x_end = x_start + CELL_WIDTH * 1000
            
            # Gradient effect for battery regions
            ax.axvspan(x_start, x_end, alpha=0.25, color=COLORS['battery'], 
                      label='Battery Cell' if i==0 else '', zorder=1)
            
            # Add vertical lines for clarity
            ax.axvline(x_start, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(x_end, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Reference line
        ax.axhline(y=27, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                  label='Initial Temperature')
        
        # Formatting
        ax.set_xlabel('Width (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
        ax.set_title(f'Temperature Profile Along Centerline - {cooling_type.title()} Cooling', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best', fontsize=13)
        
        # Set y-axis limits for better visualization
        ax.set_ylim(26, max(T_profile_smooth) + 2)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/temp_profile_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_temperature_statistics_publication(self, cooling_type='natural'):
        """Plot enhanced temperature statistics separately."""
        T_3d = self.T.value.reshape(MESH_NX, MESH_NY, MESH_NZ, order='F')
        T_celsius = T_3d - 273.15
        
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # Calculate statistics
        T_max = np.max(T_celsius)
        T_min = np.min(T_celsius)
        T_avg = np.mean(T_celsius)
        T_battery_avg = np.mean(T_celsius.flatten()[self.battery_mask])
        T_air_avg = np.mean(T_celsius.flatten()[~self.battery_mask])
        
        categories = ['Maximum', 'Battery\nAverage', 'Air\nAverage', 'Overall\nAverage', 'Minimum']
        temperatures = [T_max, T_battery_avg, T_air_avg, T_avg, T_min]
        colors = [COLORS['max_temp'], COLORS['battery'], COLORS['air_temp'], '#32CD32', '#4169E1']
        
        # Enhanced bar plot with gradient
        bars = ax.bar(categories, temperatures, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=2)
        
        # Add gradient effect to bars
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
            # Add shadow effect
            shadow = patches.Rectangle((bar.get_x() + 0.02, 0), bar.get_width(), 
                                     bar.get_height(), facecolor='gray', alpha=0.3, zorder=1)
            ax.add_patch(shadow)
        
        # Add value labels with enhanced style
        for bar, temp in zip(bars, temperatures):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{temp:.1f}°C', ha='center', va='bottom', fontweight='bold',
                   fontsize=14, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                         edgecolor='gray', alpha=0.8))
        
        # Reference line
        ax.axhline(y=27, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                  label='Initial (27°C)', zorder=5)
        
        # Formatting
        ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
        ax.set_title(f'Temperature Statistics - {cooling_type.title()} Cooling', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=12)
        
        # Set y-axis limits
        ax.set_ylim(0, max(temperatures) * 1.15)
        
        # Add background
        ax.set_facecolor('#F8F8F8')
        
        # Save figure
        filename = f'{OUTPUT_DIR}/temp_statistics_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_boundary_conditions_2d_publication(self, cooling_type='natural'):
        """Plot enhanced 2D boundary conditions with color coding."""
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Pack outline with enhanced style
        pack_rect = patches.Rectangle((0, 0), self.pack_width*1000, self.pack_depth*1000,
                                    linewidth=3, edgecolor='black', facecolor=COLORS['air'], alpha=0.3)
        ax.add_patch(pack_rect)
        
        # Enhanced battery cells without text
        for i in range(self.nx_cells):
            for j in range(self.ny_cells):
                x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS) * 1000
                x_end = x_start + CELL_WIDTH * 1000
                y_start = j * (CELL_THICKNESS + GAP_BETWEEN_CELLS) * 1000
                y_end = y_start + CELL_THICKNESS * 1000
                
                # Battery cell with gradient effect
                cell_rect = FancyBboxPatch((x_start, y_start), x_end-x_start, y_end-y_start,
                                         boxstyle="round,pad=2", linewidth=2.5, 
                                         edgecolor='darkred', facecolor=COLORS['battery'], 
                                         alpha=0.9, zorder=3)
                ax.add_patch(cell_rect)
                
                # Heat source indicator (no text)
                x_center = (x_start + x_end) / 2
                y_center = (y_start + y_end) / 2
                circle = patches.Circle((x_center, y_center), 8, color=COLORS['heat'], 
                                      alpha=0.9, zorder=4)
                ax.add_patch(circle)
        
        # Enhanced boundary visualization with color-coded walls
        wall_thickness = 8
        
        # Top wall (insulated) - blue
        top_wall = patches.Rectangle((-wall_thickness, self.pack_depth*1000), 
                                   self.pack_width*1000 + 2*wall_thickness, wall_thickness,
                                   facecolor=COLORS['insulated'], edgecolor='black', 
                                   linewidth=2, alpha=0.8, zorder=5)
        ax.add_patch(top_wall)
        
        # Bottom wall (insulated) - blue
        bottom_wall = patches.Rectangle((-wall_thickness, -wall_thickness), 
                                      self.pack_width*1000 + 2*wall_thickness, wall_thickness,
                                      facecolor=COLORS['insulated'], edgecolor='black', 
                                      linewidth=2, alpha=0.8, zorder=5)
        ax.add_patch(bottom_wall)
        
        # Left wall (convection) - orange-red
        left_wall = patches.Rectangle((-wall_thickness, 0), 
                                    wall_thickness, self.pack_depth*1000,
                                    facecolor=COLORS['convection'], edgecolor='black', 
                                    linewidth=2, alpha=0.8, zorder=5)
        ax.add_patch(left_wall)
        
        # Right wall (convection) - orange-red
        right_wall = patches.Rectangle((self.pack_width*1000, 0), 
                                     wall_thickness, self.pack_depth*1000,
                                     facecolor=COLORS['convection'], edgecolor='black', 
                                     linewidth=2, alpha=0.8, zorder=5)
        ax.add_patch(right_wall)
        
        # Enhanced convection arrows with better styling
        arrow_props = dict(arrowstyle='->', lw=3.5, color=COLORS['convection'])
        arrow_length = 25
        
        # Side wall arrows
        for i in range(5):
            x_pos = (i + 0.5) * self.pack_width * 1000 / 5
            # Top arrows
            ax.annotate('', xy=(x_pos, self.pack_depth*1000 + wall_thickness + arrow_length), 
                       xytext=(x_pos, self.pack_depth*1000 + wall_thickness + 5), 
                       arrowprops=arrow_props, zorder=6)
            # Bottom arrows
            ax.annotate('', xy=(x_pos, -wall_thickness - arrow_length), 
                       xytext=(x_pos, -wall_thickness - 5), 
                       arrowprops=arrow_props, zorder=6)
        
        for j in range(3):
            y_pos = (j + 0.5) * self.pack_depth * 1000 / 3
            # Left arrows
            ax.annotate('', xy=(-wall_thickness - arrow_length, y_pos), 
                       xytext=(-wall_thickness - 5, y_pos), 
                       arrowprops=arrow_props, zorder=6)
            # Right arrows
            ax.annotate('', xy=(self.pack_width*1000 + wall_thickness + arrow_length, y_pos), 
                       xytext=(self.pack_width*1000 + wall_thickness + 5, y_pos), 
                       arrowprops=arrow_props, zorder=6)
        
        # Add legend for boundary conditions
        legend_elements = [
            patches.Patch(facecolor=COLORS['convection'], edgecolor='black', alpha=0.8, 
                         label=f'Convection (h={H_NATURAL if cooling_type=="natural" else H_FORCED if cooling_type=="forced" else H_LIQUID} W/m²K)'),
            patches.Patch(facecolor=COLORS['insulated'], edgecolor='black', alpha=0.8, 
                         label='Insulated'),
            patches.Patch(facecolor=COLORS['battery'], edgecolor='darkred', alpha=0.9, 
                         label='Battery Cell'),
            patches.Circle((0, 0), 1, color=COLORS['heat'], alpha=0.9, 
                          label='Heat Source')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        # Formatting
        ax.set_xlim(-wall_thickness - arrow_length - 10, self.pack_width*1000 + wall_thickness + arrow_length + 10)
        ax.set_ylim(-wall_thickness - arrow_length - 10, self.pack_depth*1000 + wall_thickness + arrow_length + 10)
        ax.set_aspect('equal')
        ax.set_xlabel('Width (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Depth (mm)', fontsize=16, fontweight='bold')
        ax.set_title(f'Boundary Conditions (Top View) - {cooling_type.title()} Cooling', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=1)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/boundary_conditions_2d_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_3d_temperature_publication(self, cooling_type='natural'):
        """Plot enhanced publication-quality 3D temperature distribution."""
        T_3d = self.T.value.reshape(MESH_NX, MESH_NY, MESH_NZ, order='F')
        T_celsius = T_3d - 273.15
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays
        x = np.linspace(0, self.pack_width*1000, MESH_NX)
        y = np.linspace(0, self.pack_depth*1000, MESH_NY)
        z = np.linspace(0, self.pack_height*1000, MESH_NZ)
        
        # Sample for performance
        skip = 3
        x_sample = x[::skip]
        y_sample = y[::skip]
        z_sample = z[::skip]
        T_sample = T_celsius[::skip, ::skip, ::skip]
        
        X, Y, Z = np.meshgrid(x_sample, y_sample, z_sample, indexing='ij')
        
        # Enhanced temperature scatter with better colormap
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                           c=T_sample.flatten(), cmap='hot_r', alpha=0.8, s=30,
                           edgecolors='none', vmin=T_celsius.min(), vmax=T_celsius.max())
        
        # Enhanced battery outlines
        self._add_enhanced_battery_outlines_3d(ax)
        
        # Professional colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=25, pad=0.08)
        cbar.set_label('Temperature (°C)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Enhanced formatting
        ax.set_xlabel('Width (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('Depth (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_zlabel('Height (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title(f'3D Temperature Distribution - {cooling_type.title()} Cooling', 
                    fontsize=18, fontweight='bold', pad=25)
        
        # Clean background with subtle grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=25, azim=45)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/temp_3d_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_3d_boundaries_publication(self, cooling_type='natural'):
        """Plot enhanced publication-quality 3D boundary conditions."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define pack vertices
        vertices = [
            [0, 0, 0], [self.pack_width*1000, 0, 0], 
            [self.pack_width*1000, self.pack_depth*1000, 0], [0, self.pack_depth*1000, 0],
            [0, 0, self.pack_height*1000], [self.pack_width*1000, 0, self.pack_height*1000],
            [self.pack_width*1000, self.pack_depth*1000, self.pack_height*1000], 
            [0, self.pack_depth*1000, self.pack_height*1000]
        ]
        
        # Define faces with clear color coding
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom (insulated)
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top (insulated)
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front (convection)
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back (convection)
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left (convection)
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right (convection)
        ]
        
        # Professional color scheme matching 2D plot
        colors = [COLORS['insulated'], COLORS['insulated'], 
                 COLORS['convection'], COLORS['convection'], 
                 COLORS['convection'], COLORS['convection']]
        alphas = [0.5, 0.5, 0.7, 0.7, 0.7, 0.7]
        
        # Draw boundary faces with enhanced style
        for face, color, alpha in zip(faces, colors, alphas):
            poly = [[vertex for vertex in face]]
            ax.add_collection3d(Poly3DCollection(poly, facecolors=color, alpha=alpha, 
                                               edgecolors='black', linewidths=1.5))
        
        # Enhanced battery outlines (no text)
        self._add_enhanced_battery_outlines_3d_no_text(ax)
        
        # Add enhanced heat flow arrows
        self._add_enhanced_heat_flow_arrows(ax)
        
        # Clean formatting
        ax.set_xlabel('Width (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('Depth (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_zlabel('Height (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title('3D Boundary Conditions', fontsize=18, fontweight='bold', pad=25)
        
        # Professional legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['convection'], alpha=0.7, 
                  label=f'Convection Surfaces (h={H_NATURAL if cooling_type=="natural" else H_FORCED if cooling_type=="forced" else H_LIQUID} W/m²K)'),
            Patch(facecolor=COLORS['insulated'], alpha=0.5, 
                  label='Insulated Surfaces (Top/Bottom)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                 frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        # Clean background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/boundaries_3d_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_heat_sources_3d_publication(self, cooling_type='natural'):
        """Plot enhanced publication-quality 3D heat sources."""
        Q_3d = self.Q.value.reshape(MESH_NX, MESH_NY, MESH_NZ, order='F')
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays
        x = np.linspace(0, self.pack_width*1000, MESH_NX)
        y = np.linspace(0, self.pack_depth*1000, MESH_NY)
        z = np.linspace(0, self.pack_height*1000, MESH_NZ)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Plot significant heat sources only
        heat_threshold = np.max(Q_3d) * 0.1
        heat_mask = Q_3d > heat_threshold
        
        skip = 2
        X_sample = X[::skip, ::skip, ::skip]
        Y_sample = Y[::skip, ::skip, ::skip]
        Z_sample = Z[::skip, ::skip, ::skip]
        Q_sample = Q_3d[::skip, ::skip, ::skip]
        heat_sample = heat_mask[::skip, ::skip, ::skip]
        
        # Enhanced heat source visualization with better colormap
        scatter = ax.scatter(X_sample[heat_sample], Y_sample[heat_sample], Z_sample[heat_sample], 
                           c=Q_sample[heat_sample], cmap='YlOrRd', s=40, alpha=0.9, 
                           edgecolors='darkred', linewidth=0.8, vmin=0, vmax=Q_3d.max())
        
        # Professional colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=25, pad=0.08)
        cbar.set_label('Heat Generation (W/m³)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Enhanced battery outlines (no text)
        self._add_enhanced_battery_outlines_3d_no_text(ax)
        
        # Clean formatting
        ax.set_xlabel('Width (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('Depth (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_zlabel('Height (mm)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title('3D Heat Sources Distribution', fontsize=18, fontweight='bold', pad=25)
        
        # Clean background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Save figure
        filename = f'{OUTPUT_DIR}/heat_sources_3d_{cooling_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.show()

    def plot_all_results_publication(self, cooling_type='natural'):
        """Generate all enhanced publication-quality visualizations."""
        print(f"\n{'='*60}")
        print(f"GENERATING ENHANCED PUBLICATION FIGURES - {cooling_type.upper()} COOLING")
        print(f"{'='*60}")
        
        # Individual plots
        self.plot_temperature_history_publication(cooling_type)
        self.plot_temperature_contour_publication(cooling_type)
        self.plot_temperature_profile_publication(cooling_type)
        self.plot_temperature_statistics_publication(cooling_type)
        self.plot_boundary_conditions_2d_publication(cooling_type)
        self.plot_3d_temperature_publication(cooling_type)
        self.plot_3d_boundaries_publication(cooling_type)
        self.plot_heat_sources_3d_publication(cooling_type)
        
        print(f"All {cooling_type} cooling figures saved to {OUTPUT_DIR}/")

    # ============================================================================
    # ENHANCED HELPER METHODS
    # ============================================================================
        
    def _add_enhanced_battery_outlines_3d(self, ax):
        """Add enhanced 3D battery cell outlines with thick lines and labels."""
        for i in range(self.nx_cells):
            for j in range(self.ny_cells):
                x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS) * 1000
                x_end = x_start + CELL_WIDTH * 1000
                y_start = j * (CELL_THICKNESS + GAP_BETWEEN_CELLS) * 1000
                y_end = y_start + CELL_THICKNESS * 1000
                
                # Define enhanced vertices
                vertices = [
                    [x_start, y_start, 0], [x_end, y_start, 0],
                    [x_end, y_end, 0], [x_start, y_end, 0],
                    [x_start, y_start, self.pack_height*1000], [x_end, y_start, self.pack_height*1000],
                    [x_end, y_end, self.pack_height*1000], [x_start, y_end, self.pack_height*1000]
                ]
                
                # Enhanced edges with thicker lines
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                
                # Draw with enhanced styling
                for edge in edges:
                    points = [vertices[edge[0]], vertices[edge[1]]]
                    xs, ys, zs = zip(*points)
                    ax.plot3D(xs, ys, zs, 'k-', linewidth=3.5, alpha=0.9)

    def _add_enhanced_battery_outlines_3d_no_text(self, ax):
        """Add enhanced 3D battery cell outlines without text labels."""
        for i in range(self.nx_cells):
            for j in range(self.ny_cells):
                x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS) * 1000
                x_end = x_start + CELL_WIDTH * 1000
                y_start = j * (CELL_THICKNESS + GAP_BETWEEN_CELLS) * 1000
                y_end = y_start + CELL_THICKNESS * 1000
                
                # Define enhanced vertices
                vertices = [
                    [x_start, y_start, 0], [x_end, y_start, 0],
                    [x_end, y_end, 0], [x_start, y_end, 0],
                    [x_start, y_start, self.pack_height*1000], [x_end, y_start, self.pack_height*1000],
                    [x_end, y_end, self.pack_height*1000], [x_start, y_end, self.pack_height*1000]
                ]
                
                # Enhanced edges with thicker lines
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                
                # Draw with enhanced styling (darker for better contrast)
                for edge in edges:
                    points = [vertices[edge[0]], vertices[edge[1]]]
                    xs, ys, zs = zip(*points)
                    ax.plot3D(xs, ys, zs, color='#2C2C2C', linewidth=3.5, alpha=0.95)

    def _add_enhanced_heat_flow_arrows(self, ax):
        """Add enhanced heat flow arrows with better styling."""
        arrow_length = 25  # mm
        
        # Enhanced side wall arrows with gradient effect
        # Front wall
        for i in range(3):
            x_pos = (i + 0.5) * self.pack_width * 1000 / 3
            y_pos = 0
            z_pos = self.pack_height * 1000 / 2
            ax.quiver(x_pos, y_pos, z_pos, 0, -arrow_length, 0, 
                     color=COLORS['convection'], arrow_length_ratio=0.3, 
                     linewidth=4.5, alpha=0.9)
        
        # Back wall
        for i in range(3):
            x_pos = (i + 0.5) * self.pack_width * 1000 / 3
            y_pos = self.pack_depth * 1000
            z_pos = self.pack_height * 1000 / 2
            ax.quiver(x_pos, y_pos, z_pos, 0, arrow_length, 0, 
                     color=COLORS['convection'], arrow_length_ratio=0.3, 
                     linewidth=4.5, alpha=0.9)
        
        # Left wall
        for j in range(2):
            x_pos = 0
            y_pos = (j + 0.5) * self.pack_depth * 1000 / 2
            z_pos = self.pack_height * 1000 / 2
            ax.quiver(x_pos, y_pos, z_pos, -arrow_length, 0, 0, 
                     color=COLORS['convection'], arrow_length_ratio=0.3, 
                     linewidth=4.5, alpha=0.9)
        
        # Right wall
        for j in range(2):
            x_pos = self.pack_width * 1000
            y_pos = (j + 0.5) * self.pack_depth * 1000 / 2
            z_pos = self.pack_height * 1000 / 2
            ax.quiver(x_pos, y_pos, z_pos, arrow_length, 0, 0, 
                     color=COLORS['convection'], arrow_length_ratio=0.3, 
                     linewidth=4.5, alpha=0.9)

    def _add_enhanced_battery_rectangles(self, ax):
        """Add enhanced battery rectangles to 2D plot."""
        for i in range(self.nx_cells):
            for j in range(self.ny_cells):
                x_start = i * (CELL_WIDTH + GAP_BETWEEN_CELLS) * 1000
                x_end = x_start + CELL_WIDTH * 1000
                y_start = j * (CELL_THICKNESS + GAP_BETWEEN_CELLS) * 1000
                y_end = y_start + CELL_THICKNESS * 1000
                
                # Enhanced rectangle with thicker lines and rounded corners
                rect = FancyBboxPatch((x_start, y_start), x_end-x_start, y_end-y_start,
                                    boxstyle="round,pad=1", linewidth=3.5, 
                                    edgecolor='#2C2C2C', facecolor='none', alpha=0.95)
                ax.add_patch(rect)

# ============================================================================
# ENHANCED PUBLICATION MAIN FUNCTION
# ============================================================================

def publication_main():
    """
    Main function for enhanced publication-quality analysis.
    Runs all three cooling methods and saves individual figures.
    """
    print("="*70)
    print("ENHANCED PUBLICATION QUALITY 3D BATTERY PACK ANALYSIS")
    print("Auto-saving all figures for publication use")
    print("="*70)
    
    # Create battery pack model
    pack = PublicationBatteryPack3D(use_multiprocessing=True)
    
    # Analyze all cooling methods
    cooling_methods = [
        ('natural', H_NATURAL),
        ('forced', H_FORCED), 
        ('liquid', H_LIQUID)
    ]
    
    cooling_results = {}
    
    # Process each cooling method
    for method, h_value in cooling_methods:
        print(f"\n{'-'*50}")
        print(f"PROCESSING {method.upper()} COOLING (h={h_value} W/m²K)")
        print(f"{'-'*50}")
        
        # Solve thermal problem
        pack.solve_temperature(method, time_steps=TIME_STEPS_DEFAULT, dt=DT_DEFAULT)
        
        # Extract results
        T_max = np.max(pack.T.value) - 273.15
        T_avg_battery = np.mean(pack.T.value[pack.battery_mask]) - 273.15
        T_avg_air = np.mean(pack.T.value[~pack.battery_mask]) - 273.15
        
        cooling_results[method] = {
            'T_max': T_max,
            'T_battery': T_avg_battery,
            'T_air': T_avg_air,
            'h_value': h_value
        }
        
        print(f"Results: Tmax={T_max:.1f}°C, Tbatt_avg={T_avg_battery:.1f}°C")
        
        # Generate enhanced publication figures
        pack.plot_all_results_publication(method)
    
    # Create comprehensive comparison figure
    create_enhanced_comparison_figure(cooling_results)
    
    # Final summary
    print("\n" + "="*70)
    print("ENHANCED PUBLICATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for method in ['natural', 'forced', 'liquid']:
        print(f"  - temp_history_{method}.png")
        print(f"  - temp_contour_{method}.png")
        print(f"  - temp_profile_{method}.png")
        print(f"  - temp_statistics_{method}.png")
        print(f"  - boundary_conditions_2d_{method}.png")
        print(f"  - temp_3d_{method}.png") 
        print(f"  - boundaries_3d_{method}.png")
        print(f"  - heat_sources_3d_{method}.png")
    print(f"  - cooling_comparison.png")
    print("="*70)
    
    return cooling_results

def create_enhanced_comparison_figure(cooling_results):
    """Create enhanced publication-quality comparison figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Extract data
    methods = list(cooling_results.keys())
    h_values = [cooling_results[m]['h_value'] for m in methods]
    T_max_values = [cooling_results[m]['T_max'] for m in methods]
    T_battery_values = [cooling_results[m]['T_battery'] for m in methods]
    
    # Enhanced color scheme
    colors = ['#4169E1', '#FF8C00', '#DC143C']  # Blue, Orange, Red
    
    # Plot 1: Temperature vs Heat Transfer Coefficient with enhanced styling
    ax1.plot(h_values, T_max_values, 'o-', color=colors[2], linewidth=4, 
             markersize=12, label='Maximum Temperature', markeredgecolor='black', 
             markeredgewidth=1.5)
    ax1.plot(h_values, T_battery_values, 's-', color=colors[1], linewidth=3.5, 
             markersize=10, label='Battery Average', markeredgecolor='black', 
             markeredgewidth=1.5)
    
    # Fill areas for better visualization
    ax1.fill_between(h_values, min(T_battery_values)-1, T_max_values, 
                    alpha=0.15, color=colors[2])
    ax1.fill_between(h_values, min(T_battery_values)-1, T_battery_values, 
                    alpha=0.1, color=colors[1])
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Heat Transfer Coefficient (W/m²K)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax1.set_title('Temperature vs Heat Transfer Coefficient', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=13)
    
    # Add method labels with enhanced style
    for i, (h, T_max, method) in enumerate(zip(h_values, T_max_values, methods)):
        ax1.annotate(method.title(), (h, T_max), xytext=(15, 15), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=colors[i], alpha=0.8))
    
    # Plot 2: Temperature Reduction Comparison with enhanced bars
    T_base = T_max_values[0]  # Natural cooling as base
    reductions = [(T_base - T) / T_base * 100 for T in T_max_values]
    
    bars = ax2.bar(methods, reductions, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2, width=0.6)
    
    # Add gradient effect to bars
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
        # Add shadow
        shadow = patches.Rectangle((bar.get_x() + 0.02, 0), bar.get_width(), 
                                 bar.get_height(), facecolor='gray', alpha=0.3, zorder=1)
        ax2.add_patch(shadow)
    
    # Add value labels with enhanced style
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold',
               fontsize=14, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='gray', alpha=0.8))
    
    ax2.set_ylabel('Temperature Reduction (%)', fontsize=16, fontweight='bold')
    ax2.set_title('Cooling Performance Improvement', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)
    ax2.set_ylim(0, max(reductions) * 1.2)
    
    # Add background
    ax1.set_facecolor('#F8F8F8')
    ax2.set_facecolor('#F8F8F8')
    
    plt.tight_layout()
    
    # Save comparison figure
    filename = f'{OUTPUT_DIR}/cooling_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.show()

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Execute enhanced publication-quality analysis.
    
    This will:
    1. Run thermal analysis for all three cooling methods
    2. Generate high-resolution figures for each method (separated plots)
    3. Save all figures automatically to the output directory
    4. Create a comprehensive comparison figure
    """
    
    results = publication_main()
    
    print("\n" + "="*50)
    print("READY FOR PUBLICATION!")
    print("="*50)
    print("✓ All figures saved in high resolution (300 DPI)")
    print("✓ Enhanced professional formatting and color schemes")
    print("✓ Separated plots for better clarity")
    print("✓ Color-coded boundary conditions")
    print("✓ No text labels on battery cells")
    print("✓ Enhanced 3D visualizations")
    print("✓ Ready for academic publication")
    print("="*50)