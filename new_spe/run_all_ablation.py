import subprocess
import sys


def run_scripts():
    # 在这里按顺序填写你的文件名
    scripts = ["ablation_no_structure.py", "ablation_no_archive.py", "ablation_no_moo.py"]

    for script in scripts:
        print(f"正在启动: {script}...")
        try:
            # shell=False 是更安全的做法
            # sys.executable 获取当前正在使用的 Python 解释器路径
            result = subprocess.run([sys.executable, script], check=True)

            if result.returncode == 0:
                print(f"--- {script} 运行成功 ---\n")
        except subprocess.CalledProcessError as e:
            print(f"!!! {script} 运行出错，终止后续脚本。错误代码: {e.returncode} !!!")
            break  # 如果其中一个出错，停止运行后面的脚本


if __name__ == "__main__":
    run_scripts()