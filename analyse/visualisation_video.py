from vedo import *
from pathlib import Path
import math
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Create visualization video from VTK files')
parser.add_argument('-s', '--start', type=int, default=0, help='Start timestep (default: 0)')
parser.add_argument('-e', '--end', type=int, default=10000, help='End timestep (default: 10000)')
parser.add_argument('-vn', '--video_name', type=str, default='video_output', help='Name of video output file (default: video_output)')
parser.add_argument('-sf', '--skip_frames', type=int, default=50, help='Only take every n-th frame (default: 50)')
args = parser.parse_args()

start_timestep = args.start
end_timestep = args.end
name_video = args.video_name
skip_frames = args.skip_frames # every n-th frame, 1 for all frames

# Start timer
start_time = time.time()

# Project directories (this script lives in <project>/analyse)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# file path to load VTK files from
path = str(BASE_DIR / "output") + "/"
data = "file_"
pathcomplete = path + data

# Create video using vedo's Video class
plt = Plotter(size=(2000, 2000), bg='white', offscreen=True)

# Set fixed camera position
plt.camera.SetPosition(0, 0, 30)
plt.camera.SetFocalPoint(4, 0, 0)
plt.camera.SetViewUp(0, 1, 0)


#Create explicit axes object
axes = Axes(
    xrange=(0, 20),
    yrange=(0, 20),
    xtitle="X",
    ytitle="Y",
    axes_linewidth=2,
    grid_linewidth=2,
    c="k",
    number_of_divisions=10
)
# shifts axes down for better visibility    
axes.pos(0, 0, -2)

video_path = str(SCRIPT_DIR / f"{name_video}.mp4")
video = Video(video_path, fps=1, backend='ffmpeg')

print()
total_frames = math.ceil((end_timestep - start_timestep + 1) / skip_frames)

# Stream processing - load and process one file at a time
for idx in range(0, end_timestep - start_timestep + 1, skip_frames):
    timestep = start_timestep + idx
    
    # Load file on-demand
    p = load(f"{pathcomplete}{timestep}.vtk")
    p.point_size(80)
    # build color pallette
    # TODO wenn was schief läuft mit den Farben hier checken, weil absolut fucked
    lut = build_lut(
        [
        (0.0, "#d3ddf7ff"),
        (1.0, '#117e07ff'),  
        (2.0, '#000000')
        ],
        vmin=-0.5,
        vmax= 3.0,
        below_color='white',
        above_color='black',
        nan_color='red',
        interpolate=False,
    )
    p.cmap(lut, p.pointdata["activated"]) # color='Accent'

    plt.clear()

    # Create text showing the current timestep
    timestep_text = Text2D(
        f"Timestep: {timestep}",
        pos="top-right",
        s=3,
        c="black",
        bg="white",
        alpha=0.8,
        font="Calco")
    
    plt.show(p, timestep_text, interactive=False, resetcam=False)
    #plt.show(p, axes, timestep_text, interactive=False, resetcam=False)
    
    video.add_frame()
    
    frame_num = idx // skip_frames
    print(f"\rProcessing frame {frame_num}/{total_frames-1} ({frame_num/(total_frames-1)*100:.1f}%)", end='', flush=True)
    
    # Free memory
    del p

video.close()
plt.close()

# End timer and print results
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Video creation time:  {elapsed_time:.2f}s")
print(f"Video saved to: {video_path}")
