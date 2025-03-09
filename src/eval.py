import pexpect
import os
import time
import argparse
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate VLN agent on multiple episodes.")
parser.add_argument("--model_path", required=True, help="Path to the VLNBert model checkpoint.")
parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes to evaluate.")
args = parser.parse_args()

# Fixed parameters (match your example command)
task = "go2_matterport_vision"
history_length = 9
load_run = "2024-09-25_23-22-02"  # Adjust if needed
max_attempts = 3          # Maximum retries per episode
wait_time = 10            # Seconds to wait before retrying
activity_timeout = 360    # Seconds to wait for any output
total_timeout = 360       # Total seconds per episode (6 minutes)

# Directory for logs
log_dir = "evaluation_logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize lists to store results
successes = []
progresses = []

# Loop over episodes
for episode_index in range(args.num_episodes):
    attempts = 0
    success = False
    episode_success = False
    episode_progress = 0.0
    
    while attempts < max_attempts and not success:
        attempts += 1
        print(f"Starting attempt {attempts} for episode {episode_index}")
        
        # Construct the command
        command = (
            f"python src/evaluate.py --task={task} --history_length={history_length} "
            f"--load_run={load_run} --vlnbert_model_path={args.model_path} "
            f"--episode_index={episode_index}"
        )
        
        # Spawn the process
        child = pexpect.spawn(command)
        
        # Open log file for this attempt
        log_file_path = os.path.join(log_dir, f"episode_{episode_index}_attempt_{attempts}.log")
        with open(log_file_path, "wb") as log_file:
            child.logfile = log_file  # Log output to file
            
            start_time = time.time()
            while time.time() - start_time < total_timeout:
                try:
                    # Expect the simulation stop message
                    index_match = child.expect(
                        [
                            r"\[\d+\.\d+s\] Simulation is stopped\. The app will keep running "
                            r"with physics disabled\. Press Ctrl\+C or close the window to exit the app\."
                        ],
                        timeout=activity_timeout
                    )
                    if index_match == 0:
                        success = True
                        break
                except pexpect.TIMEOUT:
                    print(f"No output in {activity_timeout} seconds for episode {episode_index}, attempt {attempts}")
                    break
                except pexpect.EOF:
                    print(f"Process crashed for episode {episode_index}, attempt {attempts}")
                    break
            # Handle process termination
            if success:
                child.sendcontrol('c')  # Send Ctrl+C to close cleanly
                child.wait()
            else:
                child.kill(9)  # Force terminate
                child.wait()
        
        # If successful, parse the log for metrics
        if success:
            with open(log_file_path, "r") as log_file:
                log_content = log_file.read()
                # Check success
                if "Success: Reached the goal within 1 meter" in log_content:
                    episode_success = True
                elif "Failure: Did not reach the goal" in log_content or "Failure: Reached maximum steps" in log_content:
                    episode_success = False
                else:
                    episode_success = False  # Default to failure if status unclear
                
                # Extract progress
                progress_match = re.search(r"Progress: (\d+\.\d+)", log_content)
                episode_progress = float(progress_match.group(1)) if progress_match else 0.0
            
            # Log per-episode result
            with open(os.path.join(log_dir, "evaluation_results.txt"), "a") as result_file:
                status = "Success" if episode_success else "Failure"
                result_file.write(f"Episode {episode_index}: {status}, Progress: {episode_progress:.2f}\n")
            
            # Store results
            successes.append(1 if episode_success else 0)
            progresses.append(episode_progress)
            break  # Move to next episode
        elif attempts < max_attempts:
            print(f"Retrying episode {episode_index} after {wait_time} seconds")
            time.sleep(wait_time)
        else:
            print(f"Failed to complete episode {episode_index} after {max_attempts} attempts. Skipping.")
            successes.append(0)
            progresses.append(0.0)

# Compute averages
success_rate = sum(successes) / len(successes) if successes else 0.0
average_progress = sum(progresses) / len(progresses) if progresses else 0.0

# Log overall results
with open(os.path.join(log_dir, "evaluation_results.txt"), "a") as result_file:
    result_file.write(f"\nOverall Success Rate: {success_rate:.2f}\n")
    result_file.write(f"Average Progress: {average_progress:.2f}\n")

print(f"Evaluation completed. Success rate: {success_rate:.2f}, Average progress: {average_progress:.2f}")