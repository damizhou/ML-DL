"""
PCAP to MFR Converter

将 PCAP 文件转换为 MFR (Multi-level Flow Representation) 格式的 PNG 图像。

使用方法:
    python pcap_to_mfr.py --input ./pcap_data --output ./mfr_data

输入目录结构:
    pcap_data/
    ├── train/
    │   ├── class_a/*.pcap
    │   └── class_b/*.pcap
    └── test/
        ├── class_a/*.pcap
        └── class_b/*.pcap

输出目录结构:
    mfr_data/
    ├── train/
    │   ├── class_a/*.png
    │   └── class_b/*.png
    └── test/
        ├── class_a/*.png
        └── class_b/*.png
"""

import os
import argparse
import glob
import binascii
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import scapy.all as scapy
except ImportError:
    print("请安装 scapy: pip install scapy")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的替代
    def tqdm(iterable, **kwargs):
        return iterable


def read_mfr_bytes(pcap_path: str) -> str:
    """从 PCAP 文件读取并生成 MFR 十六进制字符串。

    Args:
        pcap_path: PCAP 文件路径

    Returns:
        MFR 十六进制字符串 (3200 字符 = 1600 字节)
    """
    try:
        packets = scapy.rdpcap(pcap_path)
    except Exception as e:
        print(f"读取 PCAP 失败 {pcap_path}: {e}")
        return '0' * 3200

    data = []
    for packet in packets:
        try:
            # 提取 IP 层数据
            if 'IP' not in packet:
                continue

            header = binascii.hexlify(bytes(packet['IP'])).decode()

            # 提取载荷
            try:
                payload = binascii.hexlify(bytes(packet['Raw'])).decode()
                header = header.replace(payload, '')
            except:
                payload = ''

            # 头部：160 字符 (80 字节)
            if len(header) > 160:
                header = header[:160]
            else:
                header += '0' * (160 - len(header))

            # 载荷：480 字符 (240 字节)
            if len(payload) > 480:
                payload = payload[:480]
            else:
                payload += '0' * (480 - len(payload))

            data.append((header, payload))

            # 只取前 5 个数据包
            if len(data) >= 5:
                break

        except Exception as e:
            continue

    # 不足 5 个数据包，用零填充
    while len(data) < 5:
        data.append(('0' * 160, '0' * 480))

    # 拼接所有数据
    final_data = ''
    for h, p in data:
        final_data += h + p

    return final_data


def hex_to_mfr_image(hex_string: str) -> np.ndarray:
    """将十六进制字符串转换为 40x40 MFR 矩阵。

    Args:
        hex_string: MFR 十六进制字符串

    Returns:
        40x40 uint8 numpy 数组
    """
    # 转换为字节值
    content = np.array([
        int(hex_string[i:i + 2], 16)
        for i in range(0, len(hex_string), 2)
    ])

    # 重塑为 40x40
    mfr = np.reshape(content, (40, 40))
    mfr = np.uint8(mfr)

    return mfr


def convert_pcap_to_mfr(pcap_path: str, output_path: str):
    """将单个 PCAP 文件转换为 MFR PNG 图像。

    Args:
        pcap_path: 输入 PCAP 文件路径
        output_path: 输出 PNG 文件路径
    """
    # 读取 PCAP 并生成 MFR 数据
    hex_data = read_mfr_bytes(pcap_path)

    # 转换为图像
    mfr = hex_to_mfr_image(hex_data)

    # 保存为 PNG
    im = Image.fromarray(mfr, mode='L')
    im.save(output_path)


def process_dataset(input_dir: str, output_dir: str):
    """处理整个数据集目录。

    Args:
        input_dir: 输入 PCAP 目录
        output_dir: 输出 MFR 目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 查找所有 PCAP 文件
    pcap_files = list(input_path.rglob('*.pcap'))

    if not pcap_files:
        print(f"在 {input_dir} 中未找到 PCAP 文件")
        return

    print(f"找到 {len(pcap_files)} 个 PCAP 文件")

    # 创建输出目录结构
    for pcap_file in pcap_files:
        relative_path = pcap_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.png')
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # 转换所有文件
    for pcap_file in tqdm(pcap_files, desc="转换中"):
        relative_path = pcap_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.png')

        try:
            convert_pcap_to_mfr(str(pcap_file), str(output_file))
        except Exception as e:
            print(f"转换失败 {pcap_file}: {e}")

    print(f"完成！输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='PCAP to MFR Converter')
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入 PCAP 目录'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出 MFR 目录'
    )

    args = parser.parse_args()
    process_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
