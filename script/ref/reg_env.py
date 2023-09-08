import os
import shutil


ORBIT_DIR = "/home/robot/Desktop/workspace/Omniverse/Orbit/"

# 源目录
src_dir = os.path.join(os.getcwd(), '../../', 'orbit_env')
# 目标目录
dst_dir = f"{ORBIT_DIR}source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs"

# 列出源目录中的所有文件和子目录
for item in os.listdir(src_dir):
    # 完整的源文件/目录路径
    src_path = os.path.join(src_dir, item)
    
    # 完整的目标文件/目录路径
    dst_path = os.path.join(dst_dir, item)

    # 判断是文件还是目录
    if os.path.isdir(src_path):
        # 如果是目录，我们使用 shutil.copytree
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    else:
        # 如果是文件，我们使用 shutil.copy2
        shutil.copy2(src_path, dst_path)

print("复制完成!")