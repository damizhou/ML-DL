import glob
import os
import subprocess
import sys

# Global variables (edit as needed)
RSYNC_BIN = "rsync"
SSH_BIN = "ssh"
# ssh -p 13240 root@connect.bjb2.seetacloud.com
# ssh -p 16916 root@connect.bjb2.seetacloud.com
# ssh -p 35983 root@connect.bjb2.seetacloud.com
# ssh -p 33846 root@connect.bjb1.seetacloud.com
# ssh -p 13240 root@connect.bjb2.seetacloud.com
PORT = 13240
PATTERN = "/home/pcz/DL/ML_DL/YaTC/data/vpn_top10_split/"
USER = "root"
HOST = "connect.bjb2.seetacloud.com"
REMOTE_DIR = "/root/autodl-tmp/YaTC/data/vpn_top10_split"


def main() -> int:
    files = glob.glob(PATTERN)
    if not files:
        # 注意：如果 PATTERN 是目录且没有通配符，glob 会检查该目录是否存在
        print(f"No files matched pattern: {PATTERN}")
        return 1

    dest = f"{USER}@{HOST}:{REMOTE_DIR}"
    
    mkdir_cmd = [
        SSH_BIN,
        "-p",
        str(PORT),
        f"{USER}@{HOST}",
        f"mkdir -p {REMOTE_DIR}",
    ]
    
    # --- 修改部分开始 ---
    cmd = [
        RSYNC_BIN,
        "-avz",
        "--info=progress2",  # <--- 关键修改：显示整体进度条、速度和剩余时间
        "--no-inc-recursive", # 推荐：配合 progress2 使用，为了更准确计算剩余时间（可选）
        "--rsh",
        f"{SSH_BIN} -p {PORT}",
        *files,
        dest,
    ]
    # --- 修改部分结束 ---

    try:
        print("Ensuring remote dir:", " ".join(mkdir_cmd))
        subprocess.run(mkdir_cmd, check=True)
        
        print("\nStarting Transfer...")
        print("Running:", " ".join(cmd))
        print("-" * 40)
        
        # subprocess.run 默认会将 stdout 输出到控制台，所以 rsync 的进度条会直接显示
        subprocess.run(cmd, check=True)
        
        print("\nTransfer completed successfully.")
        return 0
    except FileNotFoundError:
        print(f"{RSYNC_BIN} not found in PATH")
        return 127
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}")
        return exc.returncode


if __name__ == "__main__":
    sys.exit(main())