import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

################################################################################################################################################################
# Memory threshold in GB
MEM_THRESHOLD = 30
CUDA_INDEX = 0
# Check interval in seconds (10 minutes)
CHECK_INTERVAL = 10 * 60
AUTO_CONTINUE = False  # Set to False if you want to stop on errors instead of auto-continuing
CUDA_DEVICES = "0"  # GPUs to use for jobs (e.g. "0,1,2,3")
################################################################################################################################################################


# JOBS = [
#     {
#         "name": f"generate_{MODEL_SHORT}{_log_suffix}",
#         "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.generate_well_known_task_completions --model-key="{MODEL_NAME}" --batch-size=32 {_tasks_flag} -v 2>&1 | tee logs/generate_{MODEL_SHORT}{_log_suffix}.log''',
#     },
#     {
#         "name": f"goalscope_{MODEL_SHORT}{_log_suffix}",
#         "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.run_goalscope --model-key="{MODEL_NAME}" --batch-size=32 --overwrite {_tasks_flag} -v 2>&1 | tee logs/goalscope_{MODEL_SHORT}{_log_suffix}.log''',
#     },
#     {
#         "name": f"label_{MODEL_SHORT}_claude{_log_suffix}",
#         "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.label_well_known_task_blocks --model-key="{MODEL_NAME}" --oracle="claude-sonnet-4-6" {_tasks_flag} -v 2>&1 | tee logs/label_{MODEL_SHORT}_claude{_log_suffix}.log''',
#     },
#     {
#         "name": f"label_{MODEL_SHORT}_qwen{_log_suffix}",
#         "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.label_well_known_task_blocks --model-key="{MODEL_NAME}" --oracle="Qwen/Qwen2.5-Coder-32B-Instruct" --batch-size=32 {_tasks_flag} -v 2>&1 | tee logs/label_{MODEL_SHORT}_qwen{_log_suffix}.log''',
#     },
#     {
#         "name": f"cache_{MODEL_SHORT}{_log_suffix}",
#         "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.cache_subspace_data --model-key="{MODEL_NAME}" --oracle="claude-sonnet-4-6" {_tasks_flag} -v 2>&1 | tee logs/cache_{MODEL_SHORT}{_log_suffix}.log''',
#     }
# ]


JOBS = [
    {
        "name": "qwen14b_cma",
        "command": 'python -m scripts.clock.sweep_multiline_patch --model="Qwen/Qwen2.5-Coder-14B-Instruct" --n-pairs=100 --overwrite |& tee logs/cma_multiline_14.log '
    },
    {
        "name": "qwen32b_cma",
        "command": 'python -m scripts.clock.sweep_multiline_patch --model="Qwen/Qwen2.5-Coder-32B-Instruct" --n-pairs=100 --overwrite --quantize |& tee logs/cma_multiline_32.log '
    },
]



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
        # Wrap in `set -o pipefail` so that `python ... | tee ...` propagates
        # the Python script's exit code instead of always returning tee's 0.
        subprocess.run(
            f"set -o pipefail; {command}",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
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
            if AUTO_CONTINUE:
                print("\nJob failed. Auto-continuing to next job...")
                continue
            else:
                print("\nJob failed. Stopping execution.")
                break

    print("\n" + "=" * 80)
    print("All jobs completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
