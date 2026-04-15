import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

################################################################################################################################################################
# Memory threshold in GB
MEM_THRESHOLD = 40
CUDA_INDEX = 0
# Check interval in seconds (10 minutes)
CHECK_INTERVAL = 10 * 60
AUTO_CONTINUE = False  # Set to False if you want to stop on errors instead of auto-continuing
CUDA_DEVICES = "0,1,2,3"  # GPUs to use for jobs (e.g. "0,1,2,3")
################################################################################################################################################################


MODELS = [
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    # "facebook/cwm",
    # "allenai/Olmo-3.1-32B-Instruct",
    # # "Qwen/Qwen3.5-27B",
    # # "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct",
    # "meta-llama/Llama-3.3-70B-Instruct",
]
# JOBS = [
#     {
#         "name": f"{model}",
#         "command": f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m scripts.train_head_mask --model-key="{model}" --n-epochs=50 --batch-size=64 --augment --lazy --checkpoint-every=5 -v 2>&1 | tee logs/{model.split("/")[-1]}.log',
#     }
#     for model in MODELS
# ]

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_SHORT = "qwen-32"
JOBS = [
    {
        "name": f"generate_{MODEL_SHORT}",
        "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.generate_well_known_task_completions --model-key="{MODEL_NAME}" --batch-size=32 -v 2>&1 | tee logs/generate_{MODEL_SHORT}.log''',
    },
    {
        "name": f"goalscope_{MODEL_SHORT}",
        "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.run_goalscope --model-key="{MODEL_NAME}" --batch-size=32 --overwrite -v 2>&1 | tee logs/goalscope_{MODEL_SHORT}.log''',
    },
    {
        "name": f"label_{MODEL_SHORT}_claude",
        "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.label_well_known_task_blocks --model-key="{MODEL_NAME}" --oracle="claude-sonnet-4-6" -v 2>&1 | tee logs/label_{MODEL_SHORT}_claude.log''',
    },
    {
        "name": f"label_{MODEL_SHORT}_qwen",
        "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.label_well_known_task_blocks --model-key="{MODEL_NAME}" --oracle="Qwen/Qwen2.5-Coder-32B-Instruct" --batch-size=32 -v 2>&1 | tee logs/label_{MODEL_SHORT}_qwen.log''',
    },
    {
        "name": f"cache_{MODEL_SHORT}",
        "command": f'''CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.cache_subspace_data --model-key="{MODEL_NAME}" --oracle="claude-sonnet-4-6" -v 2>&1 | tee logs/cache_{MODEL_SHORT}.log''',
    }
]


# SUBSPACE_PROBE_CONFIGS = [
#     # {
#     #     "rank": 64,
#     #     "layer_range": (1, 11),
#     # },
#     {
#         "rank": 64,
#         "layer_range": (35, 46),
#     },
#     # {
#     #     "rank": 64,
#     #     "layer_range": (45, 56),
#     # },
#     # {
#     #     "rank": 128,
#     #     "layer_range": (1, 11),
#     # },
#     {
#         "rank": 128,
#         "layer_range": (35, 46),
#     },
#     # {
#     #     "rank": 128,
#     #     "layer_range": (45, 56),
#     # },
# ]

# # JOBS = []
# for conf in SUBSPACE_PROBE_CONFIGS:
#     rank = conf["rank"]
#     layer_start, layer_stop = conf["layer_range"]
#     job_name = f"rank={rank}__layer={layer_start}_{layer_stop}"

#     command = f"""CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python -m scripts.train_subspace_probe --model-key="Qwen/Qwen2.5-Coder-32B-Instruct" --layer-range {layer_start} {layer_stop} --r={rank} --batch-size=32 --epochs=5 2>&1 | tee logs/train_subspace/rank={rank}__layer={layer_start}_{layer_stop}.log"""
#     JOBS.append({"name": job_name, "command": command})




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
