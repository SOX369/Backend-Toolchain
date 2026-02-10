# 2_test_taskModule_controlModule.py

"""
从任务指令配置文件中提取任务信息，生成FIFO队列管理信息，并与总控配置信息合并，最终生成一个包含控制信息（总控和FIFO配置）和任务指令的完整配置文件。
"""

# 总控制器指令
total_controller_instructions = [
    "10001000111000000000000000101000111001100101001110100000000000001000011000110011000000000000000010000110000100100000000000000000",
    "11100010000100010000000000000000101100001110011100000000000000011000101010100000000000000000000011101000000101111010100000000000",
    "11000010111000000000000000001000111010000001011110101000000000001100011011100000000000000000101011000100111000000000000000000010",
    "10110100000000000000000000000000101101000000000000000000000000001011010000000000000000000000000011111100000000000000000000000000"
]

# 读取地址对齐的任务指令配置文件
task_instruction_file = "总任务指令配置_per_task_addr256k.txt"
with open(task_instruction_file, "r", encoding="utf-8") as f:
    task_lines = [line.strip() for line in f.readlines()]

# 重新分析任务指令，记录每次任务的起始行数和指令体条数
task_info = []
i = 0

while i < len(task_lines):
    # 跳过开头的全1行
    while i < len(task_lines) and task_lines[i] == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
        i += 1

    if i >= len(task_lines):
        break

    # 找到任务开始
    task_start = i

    # 寻找任务结束（寻找下一个连续的全1行或文件结束）
    consecutive_ones = 0
    while i < len(task_lines):
        if task_lines[i] == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
            consecutive_ones += 1
            if consecutive_ones >= 1:  # 遇到全1行就认为任务可能结束
                # 继续读取，直到非全1行或文件结束
                j = i + 1
                while j < len(task_lines) and task_lines[j] == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
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
            while task_end > task_start and task_lines[task_end - 1] == "conv_12x12x20_8x8x10_k5_s1_p0" * 128:
                task_end -= 1
            if task_start < task_end:
                task_info.append((task_start, task_end - task_start))

print(f"检测到 {len(task_info)} 个任务")
for idx, (start, count) in enumerate(task_info):
    final_start_line = start + 513  # 加上512行控制信息 + conv_12x12x20_8x8x10_k5_s1_p0（从1开始计数）
    address = final_start_line - 1
    print(f"任务 {idx + 1}: 地址对齐文件中第 {start + 1} 行, 最终文件中第 {final_start_line} 行, 地址 {address}, 指令条数 {count}")
    print(f"  地址是否为256倍数: {address % 256 == 0}")

# 生成FIFO信息
fifo_info = []
for start, count in task_info:
    actual_start_line = start + 513
    part1 = "0" * 64
    part2 = bin((actual_start_line - 1) * 16)[2:].zfill(32)
    part3 = bin(count)[2:].zfill(32)
    fifo_info.append(part1 + part2 + part3)

# 修改total_controller_instructions中第一行的第25至第32位为FIFO信息条数
fifo_count = len(fifo_info)
fifo_count_binary = bin(fifo_count)[2:].zfill(8)
new_first_line = total_controller_instructions[0][:24] + fifo_count_binary + total_controller_instructions[0][32:]
total_controller_instructions[0] = new_first_line


# 生成控制指令配置
control_instructions = []
control_instructions.extend(total_controller_instructions)

for _ in range(256 - len(total_controller_instructions)):
    control_instructions.append("conv_12x12x20_8x8x10_k5_s1_p0" * 128)

control_instructions.extend(fifo_info)

while len(control_instructions) < 512:
    control_instructions.append("conv_12x12x20_8x8x10_k5_s1_p0" * 128)

# 合并控制指令配置和总任务指令配置文件内容
new_lines = [line + "\n" for line in control_instructions] + [line + "\n" for line in task_lines]

# 写入新文件
new_file = "控制信息配置+总任务指令配置.txt"
with open(new_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"已生成 {new_file}，包含 {len(task_info)} 个任务的FIFO信息")

