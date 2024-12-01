import subprocess

# 定义参数组合
commands = [
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
    "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cpu --data Opennem --task_name forecasting --data_path France_processed_0.csv --target 'Fossil Gas  - Actual Aggregated [MW]'",
]



# 定义日志文件路径
log_file = "output_log.txt"

# 打开日志文件
with open(log_file, "w") as f:
    for i, command in enumerate(commands):
        print(f"正在运行第 {i + 1} 个组合：{command}")
        f.write(f"正在运行第 {i + 1} 个组合：\n{command}\n")
        try:
            # 执行命令并捕获输出
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            # 将输出写入日志文件
            f.write(f"第 {i + 1} 个组合运行成功！\n")
            f.write("输出内容如下：\n")
            f.write(result.stdout)
            f.write("\n" + "=" * 80 + "\n")
            print(f"第 {i + 1} 个组合运行成功！")
        except subprocess.CalledProcessError as e:
            # 将错误写入日志文件
            f.write(f"第 {i + 1} 个组合运行失败！\n")
            f.write("错误信息如下：\n")
            f.write(e.stderr)
            f.write("\n" + "=" * 80 + "\n")
            print(f"第 {i + 1} 个组合运行失败！")
