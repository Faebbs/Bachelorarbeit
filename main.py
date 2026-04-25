import subprocess
import sys

# Video parameters
start_timestep_video = 0
end_timestep_video = 30
skip_frames = 1
name_video = "video_output"

def compile_cuda_code():
    # Compile CUDA code using nvcc and optimization level 3
    try:
        subprocess.run(["nvcc", "-O3", "/home/fabian/Bachelorarbeit/yalla-main/foundation.cu"], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: NVCC compiler not found. Please make sure CUDA toolkit is installed.")
        return False
    
def execute_cuda_program():
    # Execute the compiled program
    args = [
        "./a.out"
        ]
    try:
        subprocess.run(args, check=True)
        print("Program executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Program execution failed with error code {e.returncode}")
        sys.exit(1)

def create_video():
    # Create video using visualisation_video.py
    try:
        subprocess.run(["python", 
                        "/home/fabian/Bachelorarbeit/analyse/visualisation_video.py", 
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
    
    execute_cuda_program()

    #create_video()