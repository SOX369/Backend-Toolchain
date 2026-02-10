# 3_test_taskModule_controlModule_dataModule.py

import os
import json
from typing import List, Dict, Tuple

"""
数据模块链接器:
从数据库中找到与网络结构匹配的算子，读取每个算子的数据文件（输入数据、权重数据、输出数据），
将这些数据按网络层序链接，为神经网络的每一层匹配对应的数据文件,生成包含完整数据信息的配置文件，并建立数据地址映射表。
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


def find_matching_db_operator(layer: Dict, db_root: str = "Data_Library") -> str:
    """在数据库中查找匹配层配置的算子目录"""
    for op_dir in os.listdir(db_root):
        full_path = os.path.join(db_root, op_dir)
        if not os.path.isdir(full_path):
            continue

        # 读取数据库算子的json（结构与算子库一致）
        json_path = os.path.join(full_path, "info.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            db_info = json.load(f)

        # 使用与1_test_taskModule.py完全相同的匹配逻辑：动态匹配条件
        match = True
        for key, value in layer.items():
            if key == "operator":
                if db_info["operator_type"] != value:
                    match = False
                    break
            elif key == "in_W" or key == "in_H" or key == "in_channels":
                if db_info["input_tensor_shape"][{"in_W": 0, "in_H": 1, "in_channels": 2}[key]] != value:
                    match = False
                    break
            elif key == "out_W" or key == "out_H" or key == "out_channels":
                if db_info["output_tensor_shape"][{"out_W": 0, "out_H": 1, "out_channels": 2}[key]] != value:
                    match = False
                    break
            elif key == "kernel":
                if db_info["kernel_size"] != list(value):
                    match = False
                    break
            elif key == "stride":
                if db_info["stride"] != [value, value]:
                    match = False
                    break

        # 检查 repeate（需要从外部传入）
        if match and "repeate" in layer and db_info.get("repeate", 1) != layer["repeate"]:
            match = False

        if match:
            return full_path
    return None

def link_data_files(op_dir: str, op_info: Dict, current_line: int) -> Tuple[List[str], List[Dict], int]:
    """按JSON字段顺序链接数据文件并生成记录"""
    separator = ["conv_12x12x20_8x8x10_k5_s1_p0" * 128 + "\n" for _ in range(5)]  # 5行分隔符
    data_content = []
    records = []

    # 提取JSON中的数据字段（保持原有顺序）
    data_fields = []
    for key in op_info.keys():
        if key in ["input_data", "weight_data", "output_data"]:
            data_fields.append(key)

    for i, field in enumerate(data_fields):
        file_path = os.path.join(op_dir, f"{field}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件缺失: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            expected_lines = op_info[field]
            if len(lines) != expected_lines:
                raise ValueError(f"{file_path}行数不符：预期{expected_lines}行，实际{len(lines)}行")

            # 确保最后一行有换行符
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'

            start_line = current_line + 1  # 确保起始行号从 conv_12x12x20_8x8x10_k5_s1_p0 开始
            data_content.extend(lines)
            records.append({
                "layer_type": op_info["operator_type"],
                "data_type": field,
                "file_path": os.path.abspath(file_path),
                "start_line": start_line,
                "lines": expected_lines
            })
            current_line += expected_lines

            # 添加字段间分隔符（非最后一个字段）
            if i != len(data_fields) - 1:
                data_content.extend(separator)
                current_line += 5

    return data_content, records, current_line

def process_network_structure(network_structure: List[Dict], task_lines: int) -> Tuple[List[str], List[Dict], Dict]:
    """处理整个网络结构，生成数据模块激励和记录，并生成data_addresses"""
    db_root = "Data_Library"
    data_incentive = []
    all_records = []
    layer_repeats = {}  # 记录每层的重复次数
    layer_separator = ["conv_12x12x20_8x8x10_k5_s1_p0" * 128 + "\n" for _ in range(5)]  # 层间分隔符
    current_line = task_lines
    data_addresses = {}

    for idx, layer in enumerate(network_structure, 1):
        # 生成层签名（用于计算repeate）
        layer_signature = (
            layer["operator"], layer["in_W"], layer["in_H"], layer["in_channels"],
            layer["out_W"], layer["out_H"], layer["out_channels"],
            layer.get("kernel"), layer.get("stride")    # 使用get方法避免KeyError
        )
        layer_repeats[layer_signature] = layer_repeats.get(layer_signature, 0) + 1

        # 创建包含repeate的层副本用于匹配
        layer_with_repeate = layer.copy()
        layer_with_repeate["repeate"] = layer_repeats[layer_signature]

        # 查找匹配的数据库算子
        op_dir = find_matching_db_operator(layer_with_repeate, db_root)
        if not op_dir:
            raise FileNotFoundError(f"未找到匹配的数据库算子：层{idx}")

        # 读取算子的json信息
        with open(os.path.join(op_dir, "info.json"), "r") as f:
            op_info = json.load(f)

        # 链接数据文件并生成记录
        content, records, current_line = link_data_files(op_dir, op_info, current_line)
        data_incentive.extend(content)
        all_records.extend([{**rec, "layer": idx} for rec in records])  # 添加层号到记录

        # 收集数据地址信息
        input_start = 0
        weight_start = 0
        output_start = 0
        weight_lines = 0
        output_lines = 0
        for record in records:
            if record["data_type"] == "input_data":
                input_start = record["start_line"]
            elif record["data_type"] == "weight_data":
                weight_start = record["start_line"]
                weight_lines = record["lines"]
            elif record["data_type"] == "output_data":
                output_start = record["start_line"]
                output_lines = record["lines"]

        # 第二层及后续层的input地址应为上一层的output地址
        if idx > 1:
            input_start = data_addresses[idx - 1]["output"]

        if layer["operator"] == "Pool":
            weight_start = 0
            weight_lines = 0

        data_addresses[idx] = {
            'input': input_start,
            'weight': weight_start,
            'output': output_start,
            'weight_lines': weight_lines,
            'output_lines': output_lines
        }

        # 添加层间分隔符（非最后一层）
        if idx < len(network_structure):
            data_incentive.extend(layer_separator)
            current_line += 5

    return data_incentive, all_records, data_addresses

def merge_with_task_instruction(task_path: str, data_incentive: List[str],
                                output_path: str = "控制信息配置+总任务指令配置+总数据信息配置.txt") -> int:
    """合并任务指令模块和数据模块，并返回任务指令的行数"""
    with open(task_path, "r", encoding="utf-8") as f:
        task_content = f.readlines()
        task_lines = len(task_content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(task_content)
        f.writelines(data_incentive)

    return task_lines             #这里返回的是 “512行的控制信息配置+总任务指令配置” 的行数

def main():
    # 沿用1_test_taskModule.py的network_structure（需包含repeate字段，由代码自动维护）

    try:
        # 从JSON文件加载网络结构配置
        network_structure = load_network_structure()

        # 合并任务指令并获取任务指令的行数，这里使用控制信息配置+总任务指令配置.txt
        task_lines = merge_with_task_instruction("控制信息配置+总任务指令配置.txt", [])

        # 处理数据模块
        data_incentive, records, data_addresses = process_network_structure(network_structure, task_lines)

        # 再次合并数据模块
        merge_with_task_instruction("控制信息配置+总任务指令配置.txt", data_incentive)

        # 打印数据记录
        print("==== 数据模块链接记录 ====")
        for record in records:
            print(f"层 {record['layer']} - {record['data_type'].upper()} 数据:")
            print(f"  文件路径: {record['file_path']}")
            print(f"  起始行号: {record['start_line']}")
            print(f"  数据行数: {record['lines']}")
            print(f"  类型: {record['layer_type']}\n")

        # 打印数据地址信息
        print("==== 数据地址信息 ====")
        print("data_addresses = {")
        for layer, info in data_addresses.items():
            print(f"    {layer}: {info},")
        print("}")

        # 保存data_addresses到JSON文件
        with open("data_addresses.json","w", encoding="utf-8") as f:
            json.dump(data_addresses,f)

    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()
