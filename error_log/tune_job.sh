#!/bin/bash
#SBATCH --job-name=tune_job            # 作业名称
#SBATCH --output=tune_output_%j.log   # 输出日志文件 (以任务ID命名)
#SBATCH --error=tune_error_%j.log     # 错误日志文件 (以任务ID命名)
#SBATCH --partition=4090              # 使用的分区 (你也可以换成 L40S)
#SBATCH --gpus=1                      # 请求1个GPU
#SBATCH --nodes=1                     # 请求1个节点
#SBATCH --time=8:00:00                # 最大运行时间 (8小时)
#SBATCH --ntasks-per-node=1           # 每节点运行1个任务

# 
# module load cuda/11.3  # 示例：加载CUDA模块，如不需要可删除

# 
source ~/.bashrc
conda activate Energy-TSF

# 
echo "Running on node: $(hostname)"
echo "CUDA devices available: $CUDA_VISIBLE_DEVICES"

# 
python tune.py
