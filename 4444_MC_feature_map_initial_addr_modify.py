# 4444_MC_feature_map_initial_addr_modify.py
import json

"""
存储控制配置地址修改
- 加载任务地址和数据地址映射文件
- 解析任务指令中的存储控制器配置（识别011开头的配置行）
- 根据数据类型（输入/权重/输出）和工作模式，修改相应的地址字段
- 将数据地址转换为27位二进制格式，拆分为高14位和低13位
- 更新存储控制器配置中的地址信息，输出最终可执行的激励文件
"""


def load_json_files(task_addresses_file, data_addresses_file):
    """加载任务地址和数据地址映射文件"""
    with open(task_addresses_file, "r", encoding="utf-8") as f:
        task_addresses = json.load(f)

    with open(data_addresses_file, "r", encoding="utf-8") as f:
        data_addresses = json.load(f)

    return task_addresses, data_addresses


def extract_bits(binary_str, start, end):
    """从二进制字符串中提取指定位范围的子串"""
    return binary_str[start:end + 1]


def replace_bits(binary_str, start, end, new_bits):
    """替换二进制字符串中指定位范围的内容"""
    return binary_str[:start] + new_bits + binary_str[end + 1:]


def addr_to_27bit_binary(addr):
    """将地址转换为27位二进制，并拆分为高14位和低13位"""
    full_addr = addr * 16  # 地址需要乘以16
    binary_27bit = format(full_addr, '027b')  # 转换为27位二进制字符串
    high_14bit = binary_27bit[:14]  # 高14位
    low_13bit = binary_27bit[14:]  # 低13位
    return high_14bit, low_13bit


def get_task_data_addresses(layer_idx, task_idx, data_addresses):
    """根据层索引和任务索引获取数据地址"""
    layer_key = f"{layer_idx}_layer"
    task_key = f"{task_idx}_task"

    if layer_key in data_addresses and task_key in data_addresses[layer_key]:
        task_data = data_addresses[layer_key][task_key]
        return {
            'input': task_data['inputData_addr'],
            'weight': task_data.get('weightData_addr', 0),  # 池化操作可能没有权重
            'output': task_data['outputData_addr']
        }
    return None


def modify_storage_controller_config(lines, start_line, task_data_addrs):
    """修改存储控制器配置中的地址字段"""
    modified_lines = lines.copy()
    current_line = start_line - 1  # 转换为0索引

    # 遍历任务指令中的所有行，寻找011开头的存储控制器配置
    task_end = current_line + 180  # 每个任务180行指令

    i = current_line
    while i < min(task_end, len(lines)):
        line = lines[i].strip()

        # 检查是否是存储控制器配置的开始（011开头）
        if len(line) == 128 and line.startswith('011'):
            # 确保有完整的三行配置
            if i + 2 < len(lines):
                line1 = lines[i].strip()
                line2 = lines[i + 1].strip()
                line3 = lines[i + 2].strip()

                # 提取关键字段
                dw = int(extract_bits(line1, 23, 24), 2)  # 数据位宽
                work_mode = int(extract_bits(line3, 113, 114), 2)  # 工作模式

                # 判断数据类型和是否需要修改
                need_modify = False
                addr_to_use = 0

                if work_mode == 0:  # DDR_TO_MC
                    if dw == 2:  # 输入数据
                        need_modify = True
                        addr_to_use = task_data_addrs['input']
                        print(f"  修改输入数据配置，地址: {addr_to_use}")
                    elif dw == 1:  # 权重数据
                        need_modify = True
                        addr_to_use = task_data_addrs['weight']
                        print(f"  修改权重数据配置，地址: {addr_to_use}")
                elif work_mode == 2:  # MC_TO_DDR
                    if dw == 2:  # 输出数据
                        need_modify = True
                        addr_to_use = task_data_addrs['output']
                        print(f"  修改输出数据配置，地址: {addr_to_use}")

                # 执行地址字段修改
                if need_modify:
                    high_14bit, low_13bit = addr_to_27bit_binary(addr_to_use)

                    # 修改line3中的initial_addr_height_14bit (50-63位)
                    modified_line3 = replace_bits(line3, 50, 63, high_14bit)

                    # 修改line3中的initial_addr_low_13bit (115-127位)
                    modified_line3 = replace_bits(modified_line3, 115, 127, low_13bit)

                    # 更新修改后的行
                    modified_lines[i + 2] = modified_line3 + '\n'

                    print(f"    原始地址: {addr_to_use}, 乘16后: {addr_to_use * 16}")
                    print(f"    27位二进制: {format(addr_to_use * 16, '027b')}")
                    print(f"    高14位: {high_14bit}, 低13位: {low_13bit}")

                # 跳过已处理的三行
                i += 3
            else:
                i += 1
        else:
            i += 1

    return modified_lines


def main(input_file, output_file, task_addresses_file, data_addresses_file):
    """主函数"""
    print("开始修改存储控制器配置中的地址字段...")

    # 加载地址映射文件
    task_addresses, data_addresses = load_json_files(task_addresses_file, data_addresses_file)

    # 读取原始配置文件
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified_lines = lines.copy()

    # 按层遍历所有任务
    global_task_counter = 1
    for layer_key in sorted(task_addresses.keys()):
        layer_idx = int(layer_key.split('_')[0])
        print(f"\n处理第{layer_idx}层:")

        for task_key in sorted(task_addresses[layer_key].keys()):
            task_idx = int(task_key.split('_')[0])
            task_info = task_addresses[layer_key][task_key]

            # 获取任务的起始行号
            actual_line = task_info['actual_line']

            # 获取任务对应的数据地址
            task_data_addrs = get_task_data_addresses(layer_idx, task_idx, data_addresses)

            if task_data_addrs is None:
                print(f"  警告: 未找到任务{task_idx}的数据地址信息")
                continue

            print(f"  任务{task_idx} (全局任务{global_task_counter}):")
            print(f"    起始行: {actual_line}")
            print(f"    输入地址: {task_data_addrs['input']}")
            print(f"    权重地址: {task_data_addrs['weight']}")
            print(f"    输出地址: {task_data_addrs['output']}")

            # 修改当前任务的存储控制器配置
            modified_lines = modify_storage_controller_config(
                modified_lines, actual_line, task_data_addrs
            )

            global_task_counter += 1

    # 写入修改后的文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)

    print(f"\n地址修改完成！输出文件: {output_file}")


if __name__ == "__main__":
    # ============ 用户配置参数 ============
    INPUT_FILE = "新版控制信息配置+总任务指令配置+总数据信息配置.txt"  # 输入：完整配置文件
    OUTPUT_FILE = "新版控制信息配置+总任务指令配置+总数据信息配置_addr_modify.txt"  # 输出：地址修改后的配置文件
    TASK_ADDRESSES_FILE = "task_addresses.json"  # 任务地址映射文件
    DATA_ADDRESSES_FILE = "data_addresses.json"  # 数据地址映射文件
    # =====================================

    main(INPUT_FILE, OUTPUT_FILE, TASK_ADDRESSES_FILE, DATA_ADDRESSES_FILE)
