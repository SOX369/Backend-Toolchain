# 1_test_taskModule.py
import os
import json

"""
两阶段任务指令配置文件生成器，主要功能是先生成原始的任务指令配置文件，然后对其进行地址对齐处理，确保每个任务的起始地址满足硬件要求（256的倍数）。
第一阶段：生成原始文件，使用固定5行分隔符
第二阶段：分析原始文件，重新生成地址对齐的文件
"""

# 常量：128-bit 全 conv_12x12x20_8x8x10_k5_s1_p0 作为算子分隔符
SEPARATOR = "1" * 128
SEPARATOR_LINES = [SEPARATOR] * 5  # 填补 5 行全 conv_12x12x20_8x8x10_k5_s1_p0

# 定义路径
base_dir = "./"
operator_library_path = os.path.join(base_dir, "Op_Library")
output_file_path = os.path.join(base_dir, "总任务指令配置.txt")
aligned_output_file_path = os.path.join(base_dir, "总任务指令配置_per_task_addr256k.txt")


network_config_path = os.path.join(base_dir, "network_structure.json")  # 网络结构配置文件路径
def load_network_structure():
    """从JSON文件加载网络结构配置"""

    with open(network_config_path, "r", encoding="utf-8") as f:
        network_structure = json.load(f)
    # 将kernel和stride从列表转换为元组（保持兼容性）
    for layer in network_structure:
        if "kernel" in layer:
            layer["kernel"] = tuple(layer["kernel"])
    return network_structure

def is_same_layer(layer1, layer2):
    """检查两个层配置是否相同"""
    return (
            layer1["operator"] == layer2["operator"] and
            layer1["in_W"] == layer2["in_W"] and
            layer1["in_H"] == layer2["in_H"] and
            layer1["in_channels"] == layer2["in_channels"] and
            layer1["out_W"] == layer2["out_W"] and
            layer1["out_H"] == layer2["out_H"] and
            layer1["out_channels"] == layer2["out_channels"] and
            layer1.get("kernel") == layer2.get("kernel") and
            layer1.get("stride") == layer2.get("stride")
    )


def find_matching_operator(layer, repeate):
    """在算子库中寻找符合 `layer` 规格，且 `repeate` 匹配的算子"""
    for operator_folder in os.listdir(operator_library_path):
        operator_path = os.path.join(operator_library_path, operator_folder)
        json_path = os.path.join(operator_path, "info.json")
        txt_path = os.path.join(operator_path, "op_jili.txt")

        if not os.path.exists(json_path) or not os.path.exists(txt_path):
            continue

        with open(json_path, "r",encoding="utf-8") as f:
            operator_info = json.load(f)

        # 动态匹配条件
        match = True
        for key, value in layer.items():
            if key == "operator":
                if operator_info["operator_type"] != value:
                    match = False
                    break
            elif key == "in_W" or key == "in_H" or key == "in_channels":
                if operator_info["input_tensor_shape"][{"in_W": 0, "in_H": 1, "in_channels": 2}[key]] != value:
                    match = False
                    break
            elif key == "out_W" or key == "out_H" or key == "out_channels":
                if operator_info["output_tensor_shape"][{"out_W": 0, "out_H": 1, "out_channels": 2}[key]] != value:
                    match = False
                    break
            elif key == "kernel":
                if operator_info["kernel_size"] != list(value):
                    match = False
                    break
            elif key == "stride":
                if operator_info["stride"] != [value, value]:
                    match = False
                    break

        if match and operator_info.get("repeate", 1) != repeate:
            match = False

        if match:
            return txt_path

    return None


def generate_original_excitation():
    """生成原始的总任务指令配置文件，按顺序组合算子（固定5行分隔）"""

    # 从JSON文件加载网络结构
    network_structure = load_network_structure()

    with open(output_file_path, "w",encoding="utf-8") as output_file:
        layer_counts = {}

        for i, layer in enumerate(network_structure):
            layer_sig = (
                layer["operator"],
                layer["in_W"], layer["in_H"], layer["in_channels"],
                layer["out_W"], layer["out_H"], layer["out_channels"],
                layer.get("kernel"), layer.get("stride")
            )

            if layer_sig in layer_counts:
                layer_counts[layer_sig] += 1
            else:
                layer_counts[layer_sig] = 1

            repeate = layer_counts[layer_sig]
            print(f"处理层 {i + 1}: {layer}，第 {repeate} 次出现此配置")

            # 查找匹配的算子
            matching_txt_path = find_matching_operator(layer, repeate)
            if not matching_txt_path:
                raise FileNotFoundError(f"没有找到匹配 {layer}，repeate = {repeate} 的算子激励")

            # 读取和写入算子激励内容
            with open(matching_txt_path, "r",encoding="utf-8") as txt_file:
                operator_content = txt_file.readlines()

            # 确保算子激励的最后一行有换行符
            if operator_content and not operator_content[-1].endswith("\n"):
                operator_content[-1] += "\n"

            # 写入算子激励内容
            output_file.writelines(operator_content)

            # 添加 5 行 128-bit 全 conv_12x12x20_8x8x10_k5_s1_p0 作为分割
            output_file.write("\n".join(SEPARATOR_LINES) + "\n")

    print(f"原始总任务指令配置文件已生成: {output_file_path}")


def find_tasks_in_original_file():
    """从原始文件中正确识别任务边界"""
    with open(output_file_path, "r",encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    tasks = []
    current_task_start = 0
    i = 0

    while i < len(lines):
        # 跳过开头的全1行
        while i < len(lines) and lines[i] == SEPARATOR:
            i += 1

        if i >= len(lines):
            break

        # 找到任务开始
        current_task_start = i

        # 寻找任务结束（连续5行全1）
        consecutive_ones = 0
        task_end = current_task_start

        j = i
        while j < len(lines):
            if lines[j] == SEPARATOR:
                consecutive_ones += 1
                if consecutive_ones >= 5:
                    # 找到了5行连续全1，任务结束
                    task_end = j - 4  # 任务结束在第一个全1行之前
                    break
            else:
                consecutive_ones = 0
            j += 1
        else:
            # 到达文件末尾
            task_end = len(lines)
            # 移除末尾可能的全1行
            while task_end > current_task_start and lines[task_end - 1] == SEPARATOR:
                task_end -= 1

        if current_task_start < task_end:
            tasks.append((current_task_start, task_end))
            print(f"找到任务 {len(tasks)}: 行 {current_task_start + 1} 到 {task_end}, 共 {task_end - current_task_start} 行")

        # 跳过分隔符，继续寻找下一个任务
        i = j + 1 if j < len(lines) else len(lines)

    return tasks, lines


def align_task_addresses():
    """处理原始文件，确保每次任务指令配置的地址为256的倍数"""
    # 从原始文件中识别所有任务
    tasks, original_lines = find_tasks_in_original_file()

    print(f"在原始文件中找到 {len(tasks)} 个任务")

    # 生成对齐后的文件
    with open(aligned_output_file_path, "w",encoding="utf-8") as output_file:
        current_line = 1  # 当前输出文件的行号（从1开始）

        for task_idx, (task_start, task_end) in enumerate(tasks):
            # 对于非第一个任务，需要确保其起始地址为256的倍数
            if task_idx > 0:
                # 计算目标地址：下一个256的倍数
                target_address = ((current_line - 1) // 256 + 1) * 256
                target_line = target_address + 1

                # 添加padding行
                padding_lines = target_line - current_line
                if padding_lines > 0:
                    for _ in range(padding_lines):
                        output_file.write(SEPARATOR + "\n")
                    current_line = target_line
                    print(f"任务 {task_idx + 1}: 添加了 {padding_lines} 行全1分隔符，从第 {current_line} 行开始，地址为 {current_line - 1}")
            else:
                print(f"任务 {task_idx + 1}: 从第 {current_line} 行开始，地址为 {current_line - 1}")

            # 写入任务内容
            task_lines_count = 0
            for line_idx in range(task_start, task_end):
                output_file.write(original_lines[line_idx] + "\n")
                task_lines_count += 1
            current_line += task_lines_count

            print(f"  任务 {task_idx + 1} 写入了 {task_lines_count} 行指令")

    print(f"地址对齐的总任务指令配置文件已生成: {aligned_output_file_path}")


def generate_final_excitation():
    """生成最终的任务指令配置文件"""
    # 第一步：生成原始文件（固定5行分隔）
    generate_original_excitation()

    # 第二步：处理地址对齐
    align_task_addresses()


if __name__ == "__main__":
    generate_final_excitation()
