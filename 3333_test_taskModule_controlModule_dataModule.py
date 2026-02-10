# 3333_test_taskModule_controlModule_dataModule.py
import os
import json
import random
from typing import List, Dict, Tuple

"""
数据模块链接
- 匹配网络结构与数据库中的算子数据文件（权重、输出数据）
- 生成第一层输入数据（随机128位二进制数据）
- 按层链接各任务所需的权重数据和输出数据
- 处理层间数据流：每层输入数据来自上一层输出数据
- 生成数据地址映射表（data_addresses.json）
- 将数据模块与任务指令文件合并，输出包含控制+任务+数据的完整配置
"""

# 常量定义
SEPARATOR = "1" * 128  # 128bit全1分隔符
SEPARATOR_LINES = [SEPARATOR + "\n"] * 5  # 5行分隔符（带换行）


def load_network_structure(network_path: str) -> List[Dict]:
    """加载网络结构配置（统一kernel为元组）"""
    with open(network_path, "r", encoding="utf-8") as f:
        network = json.load(f)
    for layer in network:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network


def calculate_input_lines(first_layer: Dict) -> int:
    """计算第一层输入数据所需行数：n = ⌈in_H / 8⌉ * in_W * in_channels"""
    in_H = first_layer["in_H"]
    in_W = first_layer["in_W"]
    in_channels = first_layer["in_channels"]
    return ((in_H + 7) // 8) * in_W * in_channels  # 向上取整


def generate_random_input(n: int) -> List[str]:
    """生成n行128bit随机01二进制数据（输入数据块）"""
    return [''.join(random.choices(['0', '1'], k=128)) + "\n" for _ in range(n)]


def match_conv_db_operator(layer, target_out_channels, operators):
    """匹配数据库中的卷积算子（与111_test_taskModule.py逻辑一致）"""
    for op in operators:
        # 基础匹配条件（算子类型、输入形状、kernel/stride等）
        if op["operator_type"] != "Conv":
            continue
        if op["input_channels"] != layer["in_channels"]:
            continue
        if op["kernel_size"] != list(layer["kernel"]):
            continue
        if op["stride"] != [layer["stride"], layer["stride"]]:
            continue
        if op.get("padding", [0, 0]) != [layer.get("padding", 0), layer.get("padding", 0)]:
            continue
        # 输出通道数精确匹配
        if op["output_channels"] != target_out_channels:
            continue
        # 匹配输入输出尺寸
        if op["input_tensor_shape"][0] != layer["in_W"]:
            continue
        if op["input_tensor_shape"][1] != layer["in_H"]:
            continue
        if op["output_tensor_shape"][0] != layer["out_W"]:
            continue
        if op["output_tensor_shape"][1] != layer["out_H"]:
            continue
        return op
    return None


def match_pool_db_operator(layer, operators):
    """匹配数据库中的池化算子（与111_test_taskModule.py逻辑一致）"""
    for op in operators:
        if op["operator_type"] != "Pool":
            continue
        if op["input_channels"] != layer["in_channels"]:
            continue
        if op["kernel_size"] != list(layer["kernel"]):
            continue
        if op["stride"] != [layer["stride"], layer["stride"]]:
            continue
        # 匹配输入输出尺寸
        if op.get("input_tensor_shape", [0, 0, 0])[0] != layer["in_W"]:
            continue
        if op.get("input_tensor_shape", [0, 0, 0])[1] != layer["in_H"]:
            continue
        if op.get("output_tensor_shape", [0, 0, 0])[0] != layer["out_W"]:
            continue
        if op.get("output_tensor_shape", [0, 0, 0])[1] != layer["out_H"]:
            continue
        if op.get("output_channels", 0) != layer["out_channels"]:
            continue
        return op
    return None


def read_db_operators(db_root: str) -> List[Dict]:
    """读取数据库中所有算子信息（含路径），类似111中的read_operator_library"""
    operators = []
    for op_dir in os.listdir(db_root):
        op_path = os.path.join(db_root, op_dir)
        if not os.path.isdir(op_path):
            continue
        # 读取算子配置（info.json）
        info_path = os.path.join(op_path, "info.json")
        if not os.path.exists(info_path):
            continue
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                op_info = json.load(f)
            # 记录算子路径（用于读取权重和输出文件）
            op_info["op_path"] = op_path
            operators.append(op_info)
        except Exception as e:
            print(f"警告：读取算子信息失败 {op_path}，错误：{str(e)}")
    return operators


def link_layer_data(layer: Dict, layer_idx: int, db_operators: List[Dict], current_line: int, task_counter: int) -> \
        Tuple[
            List[str], List[Dict], Dict, int, int]:
    """链接一层所有任务的数据（权重/输出）并记录地址"""
    data_content = []
    task_records = []  # 任务级数据记录
    layer_addresses = {}  # 层内任务地址映射：{任务索引: {input/weight/output}}

    # 确定层任务数（与111保持一致）
    if layer["operator"] == "Conv":
        total_out = layer["out_channels"]
        task_count = (total_out + 9) // 10  # 卷积按输出通道划分任务，每10个通道一个任务
    else:  # Pool
        task_count = 1  # 池化固定1个任务

    # 处理每层任务数据
    weight_lines_all = []
    for task_idx in range(task_count):
        # 匹配数据库算子（使用与111相同的匹配逻辑）
        if layer["operator"] == "Conv":
            current_out = min(10, total_out - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
        else:  # Pool
            matched_op = match_pool_db_operator(layer, db_operators)

        if not matched_op:
            # 生成详细的错误信息
            error_details = (
                f"层{layer_idx}任务{task_idx + 1}未找到匹配算子\n"
                f"网络层信息：{json.dumps(layer, indent=2)}\n"
                f"任务索引：{task_idx}"
            )
            raise FileNotFoundError(error_details)

        op_path = matched_op["op_path"]

        # 读取算子信息
        with open(os.path.join(op_path, "info.json"), "r", encoding="utf-8") as f:
            op_info = json.load(f)

        # 读取权重数据（仅卷积层）
        weight_lines = []
        if layer["operator"] == "Conv":
            weight_path = os.path.join(op_path, "weight_data.txt")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"卷积任务权重文件缺失：{weight_path}")
            with open(weight_path, "r", encoding="utf-8") as f:
                weight_lines = [line if line.endswith("\n") else line + "\n" for line in f.readlines()]
            if len(weight_lines) != op_info["weight_data"]:
                print(
                    f"警告：层{layer_idx}任务{task_idx + 1}的权重文件行数（{len(weight_lines)}）与info.json中记录的行数（{op_info['weight_data']}）不一致。")
            weight_lines_all.extend(weight_lines)

        # 记录任务数据信息
        task_records.append({
            "layer": layer_idx,
            "task": task_counter + task_idx + 1,
            "operator_type": layer["operator"],
            "weight_path": os.path.abspath(os.path.join(op_path, "weight_data.txt")) if layer[
                                                                                            "operator"] == "Conv" else None,
            "output_path": os.path.abspath(os.path.join(op_path, "output_data.txt")),
            "weight_start": 0,  # 后续更新
            "output_start": 0  # 后续更新
        })

    # 添加权重数据
    if layer["operator"] == "Conv":
        weight_start = current_line + 1
        data_content.extend(weight_lines_all)
        current_line += len(weight_lines_all)
        data_content.extend(SEPARATOR_LINES)
        current_line += 5

    # 收集所有输出数据
    output_lines_all = []
    for task_idx in range(task_count):
        # 重新匹配算子以获取输出数据
        if layer["operator"] == "Conv":
            current_out = min(10, layer["out_channels"] - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
        else:  # Pool
            matched_op = match_pool_db_operator(layer, db_operators)

        op_path = matched_op["op_path"]
        output_path = os.path.join(op_path, "output_data.txt")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"输出数据文件缺失：{output_path}")

        with open(os.path.join(op_path, "info.json"), "r", encoding="utf-8") as f:
            op_info = json.load(f)

        with open(output_path, "r", encoding="utf-8") as f:
            output_lines = [line if line.endswith("\n") else line + "\n" for line in f.readlines()]

        if len(output_lines) != op_info["output_data"]:
            print(
                f"警告：层{layer_idx}任务{task_idx + 1}的输出文件行数（{len(output_lines)}）与info.json中记录的行数（{op_info['output_data']}）不一致。")
        output_lines_all.extend(output_lines)

    # 添加输出数据
    output_start = current_line + 1
    data_content.extend(output_lines_all)
    current_line += len(output_lines_all)
    data_content.extend(SEPARATOR_LINES)
    current_line += 5

    # 更新任务记录中的权重和输出起始行
    weight_offset = weight_start if layer["operator"] == "Conv" else 0
    output_offset = output_start
    for task_idx, record in enumerate(task_records):
        if layer["operator"] == "Conv":
            # 重新匹配以获取权重信息
            current_out = min(10, layer["out_channels"] - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
            with open(os.path.join(matched_op["op_path"], "info.json"), "r", encoding="utf-8") as f:
                op_info = json.load(f)
            record["weight_start"] = weight_offset - 1  # 调整为实际行数减1
            weight_offset += op_info["weight_data"]

        # 重新匹配以获取输出信息
        if layer["operator"] == "Conv":
            current_out = min(10, layer["out_channels"] - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
        else:
            matched_op = match_pool_db_operator(layer, db_operators)

        with open(os.path.join(matched_op["op_path"], "info.json"), "r", encoding="utf-8") as f:
            op_info = json.load(f)

        record["output_start"] = output_offset - 1  # 调整为实际行数减1
        output_offset += op_info["output_data"]

    # 保存任务级地址（输入地址共享上层输出，后续统一填充）
    for task_idx in range(task_count):
        # 获取当前任务的算子信息
        if layer["operator"] == "Conv":
            current_out = min(10, layer["out_channels"] - task_idx * 10)
            matched_op = match_conv_db_operator(layer, current_out, db_operators)
        else:
            matched_op = match_pool_db_operator(layer, db_operators)

        with open(os.path.join(matched_op["op_path"], "info.json"), "r", encoding="utf-8") as f:
            op_info = json.load(f)

        layer_addresses[f"{task_counter + task_idx + 1}_task"] = {
            "inputData_addr": 0,  # 临时占位，后续更新
            "weightData_addr": task_records[task_idx]["weight_start"] if layer["operator"] == "Conv" else 0,
            "outputData_addr": task_records[task_idx]["output_start"],
            "weight_lines": op_info["weight_data"] if layer["operator"] == "Conv" else 0,
            "output_lines": op_info["output_data"]
        }

    return data_content, task_records, layer_addresses, current_line, task_counter + task_count


def process_data_module(network: List[Dict], task_file_path: str, output_path: str, db_root: str) -> Dict:
    """处理整个数据模块：生成输入数据+链接层数据+生成地址映射"""
    # 验证数据库目录是否存在
    if not os.path.exists(db_root):
        raise FileNotFoundError(f"数据库目录不存在：{os.path.abspath(db_root)}")

    # 读取数据库中所有算子信息（类似111中的操作）
    db_operators = read_db_operators(db_root)
    if not db_operators:
        raise ValueError(f"数据库中未找到有效的算子信息：{db_root}")

    # 读取任务指令文件内容（作为基础）
    with open(task_file_path, "r", encoding="utf-8") as f:
        task_content = f.readlines()
    task_lines_count = len(task_content)  # 任务指令总行数（含1536行控制信息）

    # 初始化数据内容（从任务指令末尾开始）
    data_content = []
    # 添加任务模块与数据模块的分隔符（5行全1）
    data_content.extend(SEPARATOR_LINES)
    current_line = task_lines_count + 5  # 当前行号（任务行数 + 5行分隔符）

    # 生成第一层输入数据（整个网络唯一输入）
    first_layer = network[0]
    input_lines_needed = calculate_input_lines(first_layer)
    input_data = generate_random_input(input_lines_needed)
    # 记录输入数据地址
    input_start = current_line + 1
    data_content.extend(input_data)
    current_line += input_lines_needed
    # 添加输入数据后的分隔符（5行全1）
    data_content.extend(SEPARATOR_LINES)
    current_line += 5

    # 按层处理数据（从第一层开始）
    all_records = []  # 所有数据记录
    all_addresses = {}  # 全局地址映射：{层索引: {任务索引: 地址信息}}
    prev_layer_output = input_start - 1  # 初始输入为第一层输入数据，调整为实际行数减1
    task_counter = 0  # 任务计数器

    for layer_idx, layer in enumerate(network, 1):
        print(f"处理层 {layer_idx}：{layer['operator']}（输入数据起始地址：{prev_layer_output}）")
        print(f"  层信息：in_W={layer['in_W']}, in_H={layer['in_H']}, in_channels={layer['in_channels']}")
        print(f"          out_W={layer['out_W']}, out_H={layer['out_H']}, out_channels={layer['out_channels']}")
        print(f"          kernel={layer['kernel']}, stride={layer['stride']}, padding={layer.get('padding', 0)}")

        # 链接当前层所有任务的数据（权重+输出）
        layer_data, task_records, layer_addresses, current_line, task_counter = link_layer_data(
            layer=layer,
            layer_idx=layer_idx,
            db_operators=db_operators,  # 传入预加载的数据库算子列表
            current_line=current_line,
            task_counter=task_counter
        )
        data_content.extend(layer_data)
        all_records.extend(task_records)

        # 更新当前层所有任务的输入地址（共享上层输出）
        for task_key in layer_addresses:
            layer_addresses[task_key]["inputData_addr"] = prev_layer_output

        # 记录当前层地址映射
        all_addresses[f"{layer_idx}_layer"] = layer_addresses

        # 计算当前层总输出地址（第一个任务的输出地址）
        prev_layer_output = layer_addresses[f"{task_counter - len(layer_addresses) + 1}_task"]["outputData_addr"]

    # 合并任务指令与数据模块
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(task_content)  # 任务指令（含控制信息）
        f.writelines(data_content)  # 数据模块（含分隔符）

    # 保存地址映射为JSON
    with open("data_addresses.json", "w", encoding="utf-8") as f:
        json.dump(all_addresses, f, indent=2)

    return all_addresses, all_records


def print_data_records(records: List[Dict], addresses: Dict):
    """打印数据链接记录（日志）"""
    print("\n==== 数据模块链接记录 ====")
    for record in records:
        print(f"层 {record['layer']} 任务 {record['task']}（{record['operator_type']}）:")
        if record["weight_path"]:
            print(f"  权重文件: {record['weight_path']}")
            print(f"  权重起始行: {record['weight_start']}")
        print(f"  输出文件: {record['output_path']}")
        print(f"  输出起始行: {record['output_start']}\n")

    print("==== 数据地址映射表 ====")
    print("data_addresses = {")
    for layer, tasks in addresses.items():
        print(f"  {layer}: {{")
        for task, addr in tasks.items():
            print(f"    {task}: {addr},")
        print(f"  }},")
    print("}")


def main(network_path, task_file_path, output_path, db_root):
    try:
        # 验证输入文件是否存在
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"网络结构文件不存在：{os.path.abspath(network_path)}")
        if not os.path.exists(task_file_path):
            raise FileNotFoundError(f"任务指令文件不存在：{os.path.abspath(task_file_path)}")

        # 执行数据模块处理
        network = load_network_structure(network_path)
        addresses, records = process_data_module(
            network=network,
            task_file_path=task_file_path,
            output_path=output_path,
            db_root=db_root
        )
        # 打印日志
        print_data_records(records, addresses)
        print(f"\n数据模块处理完成，输出文件：{output_path}")
        print(f"地址映射已保存：data_addresses.json")

    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    # ============ 用户配置参数 ============
    NETWORK_PATH = "network_structure.json"  # 网络结构配置文件
    TASK_FILE_PATH = "新版控制信息配置+总任务指令配置.txt"  # 输入：控制信息+任务指令文件
    OUTPUT_PATH = "新版控制信息配置+总任务指令配置+总数据信息配置.txt"  # 输出：完整配置文件
    DB_ROOT = "Data_Library"  # 数据库目录
    # =====================================

    main(NETWORK_PATH, TASK_FILE_PATH, OUTPUT_PATH, DB_ROOT)
