import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
COLOR_BG      = '#05070a'
COLOR_GROUND  = '#11151c'
COLOR_ROBOT   = '#ffb000'
COLOR_GHOSTS  = '#ff00aa'
COLOR_OBS     = '#ff3131'
COLOR_TARGET  = '#39ff14'
COLOR_TRAIL   = '#00f2ff'

OBSTACLES = [{"center": np.array([2.5, 0.0, 0.4]), "radius": 0.4}]
TARGET    = np.array([5.0, 0.0, 0.3])
BODY_DIM  = (0.4, 0.2, 0.15) # Length, Width, Height

# ─── MAIN VISUALIZER ─────────────────────────────────────────────────────────
def main():
    # 1. Load Data
    try:
        df = pd.read_csv("trajectory_log.csv")
        pos_data = df[['x', 'y', 'z']].values
        quat_data = df[['qw', 'qx', 'qy', 'qz']].values
    except FileNotFoundError:
        print("Warning: trajectory_log.csv not found. Using dummy data.")
        pos_data = np.zeros((100, 3))
        pos_data[:, 0] = np.linspace(0, 5, 100)
        pos_data[:, 2] = 0.3
        quat_data = np.tile(np.array([1, 0, 0, 0]), (100, 1))

    # Load Ghosts (if available)
    debug_rollouts = None
    if os.path.exists("debug_rollouts.bin"):
        try:
            raw_debug = np.fromfile("debug_rollouts.bin", dtype=np.float32)
            debug_rollouts = raw_debug.reshape(-1, 100, 50, 13)
        except: pass

    # 2. Setup PyVista Plotter
    plotter = pv.Plotter(off_screen=True) # Set to False if you want to see the window live
    plotter.set_background(COLOR_BG)

    # --- PERFECT DEPTH SCENE SETUP ---
    # 3. The Floor (A true 3D plane)
    floor = pv.Plane(center=(2.5, 0, 0), direction=(0, 0, 1), i_size=8, j_size=4)
    plotter.add_mesh(floor, color=COLOR_GROUND, lighting=True, show_edges=True, edge_color='#2a313d')

    # 4. Obstacles & Target
    for obs in OBSTACLES:
        sphere = pv.Sphere(radius=obs["radius"], center=obs["center"])
        plotter.add_mesh(sphere, color=COLOR_OBS, lighting=True)
        
    target_marker = pv.Sphere(radius=0.1, center=TARGET)
    plotter.add_mesh(target_marker, color=COLOR_TARGET)

    # 5. The Robot Body (Dynamic)
    robot_mesh = pv.Box(bounds=(-BODY_DIM[0]/2, BODY_DIM[0]/2, 
                                -BODY_DIM[1]/2, BODY_DIM[1]/2, 
                                -BODY_DIM[2]/2, BODY_DIM[2]/2))
    robot_actor = plotter.add_mesh(robot_mesh, color=COLOR_ROBOT, lighting=True)

    # 6. Animation Setup
    plotter.camera_position = [(2.5, -8, 6), (2.5, 0, 0), (0, 0, 1)]# Look down at an angle
    plotter.open_gif("mppi_pyvista_demo.gif")

    print("Rendering frames...")
    
    # Track trailing path
    trail_points = []

    RENDER_EVERY_N_FRAMES = 2 

    for frame in range(len(pos_data)):
        p, q = pos_data[frame], quat_data[frame]
        
        # Track trailing path (Do this every frame to keep the path accurate)
        trail_points.append(p)

        # --- SKIP RENDERING TO SAVE TIME ---
        if frame % RENDER_EVERY_N_FRAMES != 0:
            continue
        
        # A. Update Robot Pose
        rot_mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = p
        robot_actor.user_matrix = transform

        # B. Plot Trajectory Trail
        if len(trail_points) > 1:
            line = pv.lines_from_points(np.array(trail_points))
            plotter.add_mesh(line, color=COLOR_TRAIL, line_width=3, name="trail")

        # C. Plot Ghosts (Optimized)
        if debug_rollouts is not None and frame < len(debug_rollouts):
            all_lines = []
            for i in range(100):
                # Only grab the x, y, z coordinates
                ghost_pts = debug_rollouts[frame, i, :, :3]
                all_lines.append(pv.lines_from_points(ghost_pts))
            
            combined_ghosts = all_lines[0].merge(all_lines[1:])
            plotter.add_mesh(combined_ghosts, color=COLOR_GHOSTS, opacity=0.3, line_width=1, name="ghosts")

        plotter.add_text(f"MPPI SPATIAL ENGINE | T={frame*0.02:.2f}s", position='upper_left', color='white', font_size=12, name="hud")
        
        # This is the heavy function—we now call it much less often
        plotter.write_frame()

    plotter.close()
    print("Export Complete: mppi_pyvista_demo.gif")

if __name__ == "__main__":
    main()