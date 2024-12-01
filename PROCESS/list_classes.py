import os
import importlib

# 指定您要检查的文件夹路径
folder_path = 'models/TPGNN'

# 获取该文件夹中的所有 Python 文件
files = [f for f in os.listdir(folder_path) if f.endswith('.py')]

# 遍历每个文件，导入并列出其中的类
for file in files:
    module_name = f'models.TPGNN.{file[:-3]}'  # 去掉 '.py' 后缀
    module = importlib.import_module(module_name)
    classes = [cls for cls in dir(module) if isinstance(getattr(module, cls), type)]
    
    print(f'File: {file}')
    print(f'Classes: {classes}\n')
