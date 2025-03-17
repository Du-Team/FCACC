import subprocess
import os

# 定义你想要测试的参数范围
batch_sizes = [8]
learning_rates = [0.001]
pretraining_epochs = [100]
joint_optimization_epochs = [30]
w_cs = [0.2]
hard_ws = [0.2]


# 运行不同参数组合的 batch_run.py
def run_batch_run_script(batch_size, lr, pretraining_epoch, joint_optimization_epoch, w_c, hard_w):
    cmd = [
        "python", "batch_run.py",
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--pretraining_epoch", str(pretraining_epoch),
        "--MaxIter", str(joint_optimization_epoch),
        "--w_c", str(w_c),
        "--hard_w",str(hard_w)
    ]

    print(f"Running: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    log_filename = f"./log/log_batchsize_{batch_size}_lr_{lr}_pretrain_{pretraining_epoch}_finetune_{joint_optimization_epoch}_w_c_{w_c}_hard_w_{hard_w}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(stdout.decode("utf-8"))
        log_file.write("\nErrors:\n")
        log_file.write(stderr.decode("utf-8"))


for batch_size in batch_sizes:
    for lr in learning_rates:
        for pretraining_epoch in pretraining_epochs:
            for joint_optimization_epoch in joint_optimization_epochs:
                for w_c in w_cs:
                    for hard_w in hard_ws:
                        run_batch_run_script(batch_size, lr, pretraining_epoch, joint_optimization_epoch, w_c, hard_w)

print("All batch runs completed.")
os.system("/usr/bin/shutdown")
