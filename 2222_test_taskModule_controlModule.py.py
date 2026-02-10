# 2222_test_taskModule_controlModule.py.py
import json

"""
控制信息配置与FIFO管理
- 从地址对齐的任务指令文件中提取任务边界和地址信息
- 生成FIFO队列管理信息（包含任务起始地址和指令条数）
- 创建1536行控制器指令配置（前512行为总控指令，513行开始为FIFO信息）
- 将控制信息与任务指令配置合并成完整文件
- 生成并保存任务地址映射表（task_addresses.json）
"""


# 总控制器指令
total_controller_instructions = [
    "10001010111000000000000000000100111010110001011100000000000000001000100011100000000000000000101111100110011101001010110110000000",
    "10000110000100110000000000000000100001100011010000000000000000001000101001000000000000000000001011100010000100011001000000000000",
    "10110000111001110000000000000001110000001110000000000000000100101000101011100000000000000000010011101000000110001011100000000000",
    "11000011000000000000000000001100110100000000000000000000000001001011010000000000000000000000000010110100000000000000000000000000",
    "10110100000000000000000000000000101101000000000000000000000000001011010000000000000000000000000011111100000000000000000000000000"
]


def load_network_structure(network_path: str) -> list:
    """加载网络结构配置"""
    with open(network_path, "r", encoding="utf-8") as f:
        network = json.load(f)
    for layer in network:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network


def get_task_counts_per_layer(network: list) -> list:
    """根据网络结构计算每层的任务数量"""
    task_counts = []
    for layer in network:
        if layer["operator"] == "Conv":
            total_out = layer["out_channels"]
            # 与333脚本保持一致：按输出通道划分任务，每10个通道一个任务
            task_count = (total_out + 9) // 10
            task_counts.append(task_count)
        else:  # 其他算子（如Pool）固定1个任务
            task_counts.append(1)
    return task_counts


def generate_config(task_instruction_file, new_file, network_path):
    # 读取地址对齐的任务指令配置文件
    with open(task_instruction_file, "r", encoding="utf-8") as f:
        task_lines = [line.strip() for line in f.readlines()]

    # 重新分析任务指令，记录每次任务的起始行数和指令体条数
    task_info = []
    i = 0

    while i < len(task_lines):
        # 跳过开头的全1行
        while i < len(task_lines) and task_lines[i] == "1" * 128:
            i += 1

        if i >= len(task_lines):
            break

        # 找到任务开始
        task_start = i

        # 寻找任务结束（寻找下一个连续的全1行或文件结束）
        consecutive_ones = 0
        while i < len(task_lines):
            if task_lines[i] == "1" * 128:
                consecutive_ones += 1
                if consecutive_ones >= 1:  # 遇到全1行就认为任务可能结束
                    # 继续读取，直到非全1行或文件结束
                    j = i + 1
                    while j < len(task_lines) and task_lines[j] == "1" * 128:
                        j += 1

                    if j < len(task_lines):
                        # 后面还有非全1行，当前任务结束
                        task_end = i
                        task_info.append((task_start, task_end - task_start))
                        i = j
                        break
                    else:
                        # 到达文件末尾
                        task_end = i
                        task_info.append((task_start, task_end - task_start))
                        i = len(task_lines)
                        break
            else:
                consecutive_ones = 0
                i += 1

        else:
            # 循环结束，最后一个任务
            if task_start < len(task_lines):
                task_end = len(task_lines)
                # 移除末尾的全1行
                while task_end > task_start and task_lines[task_end - 1] == "1" * 128:
                    task_end -= 1
                if task_start < task_end:
                    task_info.append((task_start, task_end - task_start))

    print(f"检测到 {len(task_info)} 个任务")

    # 加载网络结构并获取每层任务数
    network = load_network_structure(network_path)
    task_counts_per_layer = get_task_counts_per_layer(network)
    print(f"从网络结构获取到 {len(task_counts_per_layer)} 层，每层任务数: {task_counts_per_layer}")

    # 生成任务指令映射表
    task_addresses = {}
    current_layer = 1
    tasks_in_current_layer = 0
    current_task_index = 0  # 全局任务索引

    # 验证任务总数是否匹配
    total_expected_tasks = sum(task_counts_per_layer)
    if len(task_info) != total_expected_tasks:
        print(f"警告: 检测到的任务数({len(task_info)})与网络结构预期的任务数({total_expected_tasks})不匹配")

    for idx, (start, count) in enumerate(task_info):
        final_start_line = start + 1536 + 1  # 加上1536行控制信息 + conv_16x16x20_12x12x10_k5_s1_p0（从1开始计数）
        address = final_start_line - 1
        print(f"任务 {idx + 1}: 地址对齐文件中第 {start + 1} 行, 最终文件中第 {final_start_line} 行, 地址 {address}, 指令条数 {count}")
        print(f"  地址是否为256倍数: {address % 256 == 0}")

        task_key = f"{idx + 1}_task"

        # 确定当前任务属于哪一层（根据网络结构定义）
        if current_layer <= len(task_counts_per_layer) and tasks_in_current_layer >= task_counts_per_layer[
            current_layer - 1]:
            current_layer += 1
            tasks_in_current_layer = 0

        # 确保不超过网络定义的层数
        if current_layer > len(task_counts_per_layer):
            layer_key = f"{current_layer}_layer"
            print(f"警告: 任务 {idx + 1} 超出网络结构定义的层数({len(task_counts_per_layer)})")
        else:
            layer_key = f"{current_layer}_layer"

        if layer_key not in task_addresses:
            task_addresses[layer_key] = {}

        # 确保每个任务只有三个固定的键值对
        task_addresses[layer_key][task_key] = {
            'actual_line': final_start_line,
            'origin_addr': address,
            'instruction_nums': count
        }

        tasks_in_current_layer += 1
        current_task_index += 1

    # 生成FIFO信息
    fifo_info = []
    for start, count in task_info:
        actual_start_line = start + 1536 + 1
        part1 = "0" * 64
        part2 = bin((actual_start_line - 1) * 16)[2:].zfill(32)
        part3 = bin(count)[2:].zfill(32)
        fifo_info.append(part1 + part2 + part3)

    # 修改total_controller_instructions中第一行的第81至第96位为FIFO信息条数
    fifo_count = len(fifo_info)
    fifo_count_binary = bin(fifo_count)[2:].zfill(16)
    new_first_line = total_controller_instructions[0][:80] + fifo_count_binary + total_controller_instructions[0][96:]
    total_controller_instructions[0] = new_first_line

    # 生成控制指令配置
    control_instructions = []
    control_instructions.extend(total_controller_instructions)

    # 填充前256 - 4行
    for _ in range(256 - len(total_controller_instructions)):
        control_instructions.append("1" * 128)

    # 继续填充到512行
    while len(control_instructions) < 512:
        control_instructions.append("1" * 128)

    # 从513行开始添加FIFO信息
    control_instructions.extend(fifo_info)

    # 继续填充到1536行
    while len(control_instructions) < 1536:
        control_instructions.append("1" * 128)

    # 合并控制指令配置和总任务指令配置文件内容
    new_lines = [line + "\n" for line in control_instructions] + [line + "\n" for line in task_lines]

    # 写入新文件
    with open(new_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"已生成 {new_file}，包含 {len(task_info)} 个任务的FIFO信息")

    # 保存任务指令映射表为JSON文件（确保保存在当前目录）
    with open("task_addresses.json", "w", encoding="utf-8") as f:
        json.dump(task_addresses, f, indent=2, ensure_ascii=False)

    # 打印任务指令映射表
    print("\ntask_addresses = {")
    # 按层号数值排序
    for layer in sorted(task_addresses.keys(), key=lambda x: int(x.split('_')[0])):
        print(f"  {layer}: {{")
        # 按任务编号数值排序
        sorted_tasks = sorted(
            task_addresses[layer].items(),
            key=lambda item: int(item[0].split('_')[0])  # 提取任务编号并转为整数排序
        )
        for task_key, data in sorted_tasks:
            data_str = ", ".join([f"'{k}': {v}" for k, v in data.items()])
            print(f"    {task_key}: {{{data_str}}},")
        print(f"  }},")
    print("}")

    return task_addresses


if __name__ == "__main__":
    # ============ 用户配置参数 ============
    NETWORK_PATH = "network_structure.json"  # 网络结构配置文件
    TASK_INSTRUCTION_FILE = "新版总任务指令配置_per_task_addr256k.txt"  # 输入：地址对齐的任务指令文件
    OUTPUT_FILE = "新版控制信息配置+总任务指令配置.txt"  # 输出：控制信息+任务指令配置文件
    # =====================================

    generate_config(TASK_INSTRUCTION_FILE, OUTPUT_FILE, NETWORK_PATH)

