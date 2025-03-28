
import submitit
from resnet import train
import argparse

# parser = argparse.ArgumentParser(description="Plot training loss from Submitit log.")
# parser.add_argument("name", type=str, help="Path to the log file")
# args = parser.parse_args()

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    slurm_time='08:00:00',  # 8 hours
    stderr_to_stdout=True,
    slurm_name="100-binary-weighted-mri-emma"
)

job = executor.submit(train, binary=True)
print(job.job_id)

output = job.result()  # waits for completion and returns output
print("done. output: ", output)