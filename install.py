"""
ComfyUI 插件自动安装脚本
"""

import subprocess
import sys
import os

def install_dependencies():
    """安装依赖包"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("未找到 requirements.txt 文件")
        return False
    
    print("正在安装 Qwen-Image-i2L 插件依赖...")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            requirements_path
        ])
        print("依赖安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        return False

if __name__ == "__main__":
    install_dependencies()
