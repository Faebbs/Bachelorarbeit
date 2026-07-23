import subprocess
import sys
from pathlib import Path

# Project root (directory this script lives in)
BASE_DIR = Path(__file__).resolve().parent

# Video parameters
start_timestep_video = 0
end_timestep_video = 30
skip_frames = 1
name_video = "video_output"

# Argumente für Simulation
Kappen_radius = 10 # 6 ist circa der kleinste sinnvolle Wert
seed = 42
mode = "vtk" # "scan" (gibt nur Werte zurück), "vtk" oder "vtk_prop" (visualisierung ohne spontante Aktivierung)
threshold = 0.175

# Welle an einer Stelle bestimmter Oberflächen-Krümmung starten (nur MODEL_EGG).
# seed_by_curvature = True  -> die Welle wird am Punkt des gewählten Krümmungs-
# Perzentils gezündet (gleiche Auswahl wie im Ei-Scan). False -> foundation.cu
# nutzt seinen Default-Seed.
seed_by_curvature = True
seed_percentile = 50   # Perzentil der elliptischen (K>0) Krümmungsverteilung: 99=spitz ... 5=flach
frac = 0.01            # Stimulusgröße (Anteil der am Seed aktivierten Zellen)


def curvature_seed(percentile):
    """(x, y, z, K, R_eq) für einen Ei-Mesh-Punkt nahe dem gegebenen Krümmungs-
    Perzentil; nutzt den Farthest-Point-Sampler des Ei-Scans (bester K-Treffer)."""
    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR / "analyse"))
    import egg_activation_scan as eas
    loc = eas.locs_at_percentile(percentile)[0]
    return loc["x"], loc["y"], loc["z"], loc["K"], loc["R_eq"]



def compile_cuda_code():
    # Compile CUDA code using nvcc and optimization level 3
    try:
        subprocess.run(["nvcc", "-O3", str(BASE_DIR / "yalla-main" / "foundation.cu"), "-o", str(BASE_DIR / "a.out")], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: NVCC compiler not found. Please make sure CUDA toolkit is installed.")
        return False
    
def execute_relaxation():
    # Egg only: settle the raw cells and write egg_cells_relaxed.vtk, which the
    # main run below then loads. Decouples the relaxation from the wave so the
    # wave starts on a settled sheet (activation stays at timestep 220).
    args = [str(BASE_DIR / "a.out"), str(Kappen_radius), str(seed), "relax", str(threshold)]
    try:
        subprocess.run(args, cwd=BASE_DIR, check=True)
        print("Relaxation completed (egg_cells_relaxed.vtk written).")
    except subprocess.CalledProcessError as e:
        print(f"Relaxation failed with error code {e.returncode}")
        sys.exit(1)

def execute_cuda_program():
    # Execute the compiled program
    args = [
        str(BASE_DIR / "a.out"), str(Kappen_radius), str(seed), mode, str(threshold)
        ]
    if seed_by_curvature:
        x, y, z, K, R_eq = curvature_seed(seed_percentile)
        print(f"Seeding wave at curvature pct{seed_percentile}: "
              f"K={K:.4f}, R_eq={R_eq:.1f} at ({x:.2f}, {y:.2f}, {z:.2f})")
        args += [str(frac), str(x), str(y), str(z)]
    try:
        # Run from the project root so the program finds its input/output files
        subprocess.run(args, cwd=BASE_DIR, check=True)
        print("Program executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Program execution failed with error code {e.returncode}")
        sys.exit(1)

def create_video():
    # Create video using visualisation_video.py
    try:
        subprocess.run(["python",
                        str(BASE_DIR / "analyse" / "visualisation_video.py"),
                        "-s", str(start_timestep_video),
                        "-e", str(end_timestep_video),
                        "-vn", name_video,
                        "-sf", str(skip_frames)],
                        check=True
                        )
        print("Video created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Video creation failed with error code {e.returncode}")

if __name__ == "__main__":
    # Compile CUDA code
    if not compile_cuda_code():
        sys.exit(1)
    else:
        print("Compilation completed successfully.")

    execute_relaxation()
    execute_cuda_program()

    #create_video()