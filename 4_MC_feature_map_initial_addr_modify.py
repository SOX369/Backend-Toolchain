# 4_MC_feature_map_initial_addr_modify.py

import json
import os
import sys  # 导入sys模块，用于接收命令行参数
from typing import Dict, List, Tuple

"""
存储配置中数据地址修改
主要功能是修改任务指令配置文件中的存储控制器配置，将feature_map_initial_addr的数据地址替换为数据块在DDR中实际的物理地址，使各任务工作指令能够正确访问对应的数据。
读取包含任务指令和数据信息的配置文件，根据数据地址映射表和网络结构，计算每个任务的实际数据地址，并修改存储控制器配置中的地址字段。
"""

# 定义路径
network_config_path = "network_structure.json"

def load_network_structure():
    """从JSON文件加载网络结构配置"""

    with open(network_config_path, "r", encoding="utf-8") as f:
        network_structure = json.load(f)
    # 将kernel从列表转换为元组（保持兼容性）
    for layer in network_structure:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network_structure


def get_layer_task_count(layer: Dict) -> int:
    """计算一层需要多少次任务指令配置"""
    if layer["operator"] == "Pool":
        return 1
    elif layer["operator"] == "Conv":
        out_channels = layer["out_channels"]
        return (out_channels + 3) // 4
    else:
        return 1


def modify_storage_controller_config(line1: str, line2: str, new_addr: int) -> Tuple[str, str]:
    """修改存储控制器配置中的feature_map_initial_addr字段"""

    def replace_bits(binary: str, start: int, end: int, value: int) -> str:
        prefix = binary[:start]
        binary_value = bin(value)[2:].zfill(end - start + 1)
        suffix = binary[end + 1:]
        return prefix + binary_value + suffix

    modified_line1 = replace_bits(line1, 73, 104, new_addr)
    return modified_line1, line2


def extract_bits(binary: str, start: int, end: int) -> int:
    """提取二进制子串并转换为整数"""
    sub = binary[start:end + 1]
    return int(sub, 2) if sub else 0


def find_task_boundaries(lines: List[str]) -> List[Tuple[int, int]]:
    """找到任务边界，支持可变长度的全1分隔符"""
    task_boundaries = []
    current_task_start = 512
    i = 512

    while i < len(lines):
        if lines[i].strip().startswith(("001", "100", "011")):
            current_task_start = i

            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
                    consecutive_ones = 0
                    k = j
                    while k < len(lines) and lines[k].strip() == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
                        consecutive_ones += 1
                        k += 1

                    if consecutive_ones > 0:
                        task_boundaries.append((current_task_start, j))
                        i = k
                        break
                j += 1
            else:
                task_boundaries.append((current_task_start, len(lines)))
                break
        else:
            i += 1

    return task_boundaries


def generate_enhanced_version(output_file):  # 新增output_file参数，用于接收输出路径
    """生成增强版本，包含实际的数据地址计算"""
    input_file = "控制信息配置+总任务指令配置+总数据信息配置.txt"
    # output_file = "控制信息配置+总任务指令配置+总数据信息配置_new.txt"

    # 从JSON文件加载网络结构配置
    network_structure = load_network_structure()

    # 从json文件中读取data_addresses
    with open("data_addresses.json", "r", encoding="utf-8") as f:
        data_addresses = json.load(f)
    data_addresses = {int(k): v for k, v in data_addresses.items()}

    print("数据地址映射表:")
    for layer_num, addr in data_addresses.items():
        print(f"层 {layer_num}: {addr}")

    # 读取并处理文件
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    modified_lines = lines.copy()

    # 找到所有任务边界
    task_boundaries = find_task_boundaries(lines)
    print(f"找到 {len(task_boundaries)} 个任务边界")

    for idx, (start, end) in enumerate(task_boundaries):
        address = start - 1
        print(f"任务 {idx + 1}: 起始行 {start + 1}, 结束行 {end}, 地址 {address}, 地址是否为256倍数: {address % 256 == 0}")

    task_index = 0
    for layer_idx, layer in enumerate(network_structure):
        layer_num = layer_idx + 1
        task_count = get_layer_task_count(layer)

        for task_in_layer in range(task_count):
            if task_index >= len(task_boundaries):
                print(f"警告：任务索引超出边界")
                break

            task_start, task_end = task_boundaries[task_index]
            print(f"处理层 {layer_num}, 任务 {task_in_layer + 1}/{task_count}, 行范围 {task_start + 1}-{task_end}")

            # 处理存储控制器配置
            i = task_start
            while i < task_end:
                if i + 1 >= len(lines):
                    break

                line1 = lines[i].strip()
                line2 = lines[i + 1].strip()

                if line1.startswith("011") and line2.startswith("011"):
                    wr_bits = extract_bits(line2, 73, 74)
                    dw = extract_bits(line2, 48, 52)

                    new_addr = 0
                    addr_type = ""

                    if wr_bits == 0:  # DDR_TO_MC
                        if dw == 16:  # 输入数据
                            if layer_idx > 0:
                                prev_layer_addr = data_addresses[layer_idx]
                                new_addr = (prev_layer_addr["output"] - 1) * 16
                            else:
                                new_addr = (data_addresses[layer_num]["input"] - 1) * 16
                            addr_type = "输入"

                        elif dw == 8:  # 权重数据
                            if data_addresses[layer_num]["weight"] > 0:
                                weight_start = data_addresses[layer_num]["weight"]
                                weight_lines = data_addresses[layer_num]["weight_lines"]

                                if task_count > 1:
                                    lines_per_task = weight_lines // task_count
                                    task_weight_start = weight_start + task_in_layer * lines_per_task
                                    new_addr = (task_weight_start - 1) * 16
                                else:
                                    new_addr = (weight_start - 1) * 16
                            addr_type = "权重"

                    elif wr_bits == 2:  # MC_TO_DDR
                        if dw == 16:  # 输出数据
                            output_start = data_addresses[layer_num]["output"]
                            output_lines = data_addresses[layer_num]["output_lines"]

                            if task_count > 1:
                                lines_per_task = output_lines // task_count
                                task_output_start = output_start + task_in_layer * lines_per_task
                                new_addr = (task_output_start - 1) * 16
                            else:
                                new_addr = (output_start - 1) * 16
                            addr_type = "输出"

                    if new_addr > 0:
                        modified_line1, _ = modify_storage_controller_config(line1, line2, new_addr)
                        modified_lines[i] = modified_line1 + '\n'
                        print(f"  修改{addr_type}数据配置: 地址 {new_addr}")

                    i += 2
                else:
                    i += 1

            task_index += 1

    # 保存结果
    with open(output_file, 'w', encoding="utf-8") as f:
        f.writelines(modified_lines)

    print(f"修改完成，结果保存到 {output_file}")


if __name__ == "__main__":
    # 新增：从命令行参数获取输出路径（前端传递的路径将在这里接收）
    if len(sys.argv) != 2:
        print("使用命令行参数方式执行该脚本：python 4_MC_feature_map_initial_addr_modify.py <输出文件路径>")
        sys.exit(1)
    output_path = sys.argv[1]  # 接收前端传入的路径
    generate_enhanced_version(output_path)
