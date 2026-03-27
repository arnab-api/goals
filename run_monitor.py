import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODELS = [
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    # "facebook/cwm",
    # "allenai/Olmo-3.1-32B-Instruct",
    # # "Qwen/Qwen3.5-27B",
    # # "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]
JOBS = [
    {
        "name": f"{model}",
        "command": f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m scripts.train_head_mask --model-key="{model}" --n-epochs=50 --batch-size=64 --augment --lazy --checkpoint-every=5 -v 2>&1 | tee logs/{model.split("/")[-1]}.log',
    }
    for model in MODELS
]

################################################################################################################################################################
# Memory threshold in GB
MEM_THRESHOLD = 40
CUDA_INDEX = 0

# Check interval in seconds (10 minutes)
CHECK_INTERVAL = 10 * 60


def get_gpu_free_memory():
    """Get free memory on cuda:0 in GB."""
    try:
        # Run nvidia-smi and capture output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_list = result.stdout.splitlines()

        # Parse the output to get the free memory in MB
        free_memory_mb = float(gpu_list[CUDA_INDEX].strip())

        # Convert to GB
        free_memory_gb = free_memory_mb / 1024.0

        return free_memory_gb
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0


def wait_for_gpu_memory():
    """Wait until GPU has enough free memory."""
    print(
        f"Waiting for cuda:{CUDA_INDEX} to have more than {MEM_THRESHOLD}GB free memory."
    )
    print(f"Will check every {CHECK_INTERVAL / 60} minutes.")

    while True:
        free_memory = get_gpu_free_memory()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] Free GPU memory: {free_memory:.2f}GB")

        if free_memory > MEM_THRESHOLD:
            print(
                f"GPU has {free_memory:.2f}GB of free memory, which exceeds threshold of {MEM_THRESHOLD}GB."
            )
            return free_memory
        else:
            print(
                f"Not enough GPU memory available. Waiting {CHECK_INTERVAL / 60} minutes before checking again..."
            )
            time.sleep(CHECK_INTERVAL)


def run_job(job_info, job_num, total_jobs):
    """Run a single job."""
    job_name = job_info["name"]
    command = job_info["command"]

    print("\n" + "=" * 80)
    print(f"JOB {job_num}/{total_jobs}: {job_name}")
    print("=" * 80)
    print(f"Command: {command}")
    print("=" * 80)

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ Job {job_num}/{total_jobs} ({job_name}) completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running job {job_num}/{total_jobs} ({job_name}): {e}")
        return False


def main():
    print(">>> Running GPU Job Queue <<<")
    print(f"Total jobs in queue: {len(JOBS)}")
    print(f"GPU threshold: {MEM_THRESHOLD}GB on cuda:{CUDA_INDEX}")
    print(f"Check interval: {CHECK_INTERVAL / 60} minutes\n")

    for i, job in enumerate(JOBS, 1):
        print(f"\n{'=' * 80}")
        print(f"Preparing to run job {i}/{len(JOBS)}: {job['name']}")
        print(f"{'=' * 80}")

        # Wait for GPU memory to be available
        wait_for_gpu_memory()

        # Run the job
        success = run_job(job, i, len(JOBS))

        if not success:
            print("\nJob failed. Auto-continuing to next job...")
            continue

    print("\n" + "=" * 80)
    print("All jobs completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
