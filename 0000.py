import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import io
import os
from datetime import datetime

# 导入你的阶段模块
try:
    import stage1_task_generator
    import stage2_control_generator
    import stage3_data_linker
    import stage4_address_modifier
    import stage0_onnx_to_json
except ImportError as e:
    print(f"警告: 无法导入模块 - {e}")


class ModernToolchainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("神经网络硬件加速器工具链")
        self.root.geometry("1400x900")

        # 配置变量
        self.onnx_model_path = tk.StringVar()
        self.network_json_path = tk.StringVar()
        self.op_library_path = tk.StringVar()
        self.data_library_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="pipeline_output")

        # 状态变量
        self.is_running = False
        self.current_stage = tk.StringVar(value="就绪")

        # 设置样式
        self.setup_styles()

        # 创建界面
        self.create_widgets()

    def setup_styles(self):
        """设置现代化样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 配置颜色方案
        bg_color = '#1e1e2e'
        fg_color = '#ffffff'
        accent_color = '#8b5cf6'
        secondary_color = '#4c1d95'

        # 主窗口背景
        self.root.configure(bg=bg_color)

        # Frame样式
        style.configure('Card.TFrame', background='#2a2a3e', relief='raised')
        style.configure('Main.TFrame', background=bg_color)

        # Label样式
        style.configure('Title.TLabel', background=bg_color, foreground=fg_color,
                        font=('Arial', 24, 'bold'))
        style.configure('Subtitle.TLabel', background=bg_color, foreground='#a78bfa',
                        font=('Arial', 11))
        style.configure('Header.TLabel', background='#2a2a3e', foreground=fg_color,
                        font=('Arial', 12, 'bold'))
        style.configure('Normal.TLabel', background='#2a2a3e', foreground='#e0e0e0',
                        font=('Arial', 10))
        style.configure('Stage.TLabel', background='#2a2a3e', foreground='#a78bfa',
                        font=('Arial', 9))

        # Button样式
        style.configure('Accent.TButton', background=accent_color, foreground=fg_color,
                        font=('Arial', 11, 'bold'), borderwidth=0)
        style.map('Accent.TButton',
                  background=[('active', secondary_color), ('disabled', '#4a4a5e')])

        style.configure('File.TButton', background='#374151', foreground=fg_color,
                        font=('Arial', 9), borderwidth=0)
        style.map('File.TButton', background=[('active', '#4b5563')])

        # Entry样式
        style.configure('Modern.TEntry', fieldbackground='#374151',
                        foreground=fg_color, borderwidth=1)

    def create_widgets(self):
        """创建所有界面组件"""
        # 主容器
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题区域
        self.create_header(main_frame)

        # 内容区域 - 使用PanedWindow分割
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=20)

        # 左侧配置面板
        left_panel = self.create_left_panel(content_paned)
        content_paned.add(left_panel, weight=1)

        # 右侧日志面板
        right_panel = self.create_right_panel(content_paned)
        content_paned.add(right_panel, weight=2)

    def create_header(self, parent):
        """创建标题区域"""
        header_frame = ttk.Frame(parent, style='Main.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame, text="🔧 神经网络硬件加速器工具链",
                                style='Title.TLabel')
        title_label.pack()

        subtitle_label = ttk.Label(header_frame,
                                   text="ONNX模型 → 硬件可执行激励文件",
                                   style='Subtitle.TLabel')
        subtitle_label.pack()

    def create_left_panel(self, parent):
        """创建左侧配置面板"""
        panel = ttk.Frame(parent, style='Card.TFrame', relief='raised', borderwidth=2)

        # 使用滚动框架以防内容过多
        canvas = tk.Canvas(panel, bg='#2a2a3e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 内部容器
        container = ttk.Frame(scrollable_frame, style='Card.TFrame', padding=20)
        container.pack(fill=tk.BOTH, expand=True)

        # 配置区域标题
        config_header = ttk.Label(container, text="📁 配置选择", style='Header.TLabel')
        config_header.pack(anchor=tk.W, pady=(0, 20))

        # 文件选择区域
        self.create_file_selectors(container)

        # 分隔线
        separator = ttk.Separator(container, orient='horizontal')
        separator.pack(fill=tk.X, pady=20)

        # 进度显示区域
        self.create_progress_section(container)

        # 分隔线
        separator2 = ttk.Separator(container, orient='horizontal')
        separator2.pack(fill=tk.X, pady=20)

        # 执行按钮 - 放在最后
        self.create_execute_button(container)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return panel

    def create_file_selectors(self, parent):
        """创建文件选择器"""
        configs = [
            ("ONNX模型文件", self.onnx_model_path, self.select_onnx_file, "file"),
            ("网络结构JSON", self.network_json_path, self.select_json_file, "file"),
            ("算子库目录", self.op_library_path, self.select_op_library, "dir"),
            ("数据库目录", self.data_library_path, self.select_data_library, "dir"),
            ("输出目录", self.output_dir, None, "entry")
        ]

        for label_text, var, command, type_ in configs:
            frame = ttk.Frame(parent, style='Card.TFrame')
            frame.pack(fill=tk.X, pady=8)

            label = ttk.Label(frame, text=label_text, style='Normal.TLabel')
            label.pack(anchor=tk.W)

            entry_frame = ttk.Frame(frame, style='Card.TFrame')
            entry_frame.pack(fill=tk.X, pady=(5, 0))

            entry = ttk.Entry(entry_frame, textvariable=var, width=30,
                              style='Modern.TEntry', font=('Arial', 9))
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            if type_ != "entry":
                btn = ttk.Button(entry_frame, text="浏览", command=command,
                                 style='File.TButton', width=8)
                btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_progress_section(self, parent):
        """创建进度显示区域"""
        progress_header = ttk.Label(parent, text="📊 执行进度", style='Header.TLabel')
        progress_header.pack(anchor=tk.W, pady=(0, 10))

        # 进度条框架
        self.progress_frame = ttk.Frame(parent, style='Card.TFrame')
        self.progress_frame.pack(fill=tk.X)

        stages = [
            ("阶段0", "ONNX模型解析"),
            ("阶段1", "任务指令生成"),
            ("阶段2", "控制信息配置"),
            ("阶段3", "数据模块链接"),
            ("阶段4", "地址修正")
        ]

        self.stage_labels = {}
        self.stage_indicators = {}

        for i, (stage, desc) in enumerate(stages):
            stage_frame = ttk.Frame(self.progress_frame, style='Card.TFrame')
            stage_frame.pack(fill=tk.X, pady=5)

            # 状态指示器
            indicator_canvas = tk.Canvas(stage_frame, width=20, height=20,
                                         bg='#2a2a3e', highlightthickness=0)
            indicator_canvas.pack(side=tk.LEFT, padx=(0, 10))

            circle = indicator_canvas.create_oval(5, 5, 15, 15,
                                                  fill='#4a4a5e', outline='#6b7280')
            self.stage_indicators[f'stage{i}'] = (indicator_canvas, circle)

            # 阶段信息
            info_frame = ttk.Frame(stage_frame, style='Card.TFrame')
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            stage_label = ttk.Label(info_frame, text=stage, style='Normal.TLabel')
            stage_label.pack(anchor=tk.W)

            desc_label = ttk.Label(info_frame, text=desc, style='Stage.TLabel')
            desc_label.pack(anchor=tk.W)

            self.stage_labels[f'stage{i}'] = stage_label

    def create_execute_button(self, parent):
        """创建执行按钮"""
        button_frame = ttk.Frame(parent, style='Card.TFrame')
        button_frame.pack(fill=tk.X, pady=(20, 0), side=tk.BOTTOM)

        # 状态标签放在按钮上方
        self.status_label = ttk.Label(button_frame, textvariable=self.current_stage,
                                      style='Stage.TLabel')
        self.status_label.pack(pady=(0, 10))

        self.execute_btn = ttk.Button(button_frame, text="▶ 开始执行",
                                      command=self.start_execution,
                                      style='Accent.TButton')
        self.execute_btn.pack(fill=tk.X, ipady=15)

    def create_right_panel(self, parent):
        """创建右侧日志面板"""
        panel = ttk.Frame(parent, style='Card.TFrame', relief='raised', borderwidth=2)

        # 内部容器
        container = ttk.Frame(panel, style='Card.TFrame', padding=20)
        container.pack(fill=tk.BOTH, expand=True)

        # 日志区域标题
        log_header = ttk.Label(container, text="📄 执行日志", style='Header.TLabel')
        log_header.pack(anchor=tk.W, pady=(0, 10))

        # 日志显示区域
        self.log_text = scrolledtext.ScrolledText(
            container,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#1a1a2e',
            fg='#e0e0e0',
            insertbackground='white',
            relief='flat',
            borderwidth=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 配置标签颜色
        self.log_text.tag_config('info', foreground='#a0a0a0')
        self.log_text.tag_config('success', foreground='#10b981')
        self.log_text.tag_config('error', foreground='#ef4444')
        self.log_text.tag_config('warning', foreground='#f59e0b')
        self.log_text.tag_config('stage', foreground='#fbbf24', font=('Consolas', 9, 'bold'))
        self.log_text.tag_config('timestamp', foreground='#6b7280')

        # 清除日志按钮
        clear_btn = ttk.Button(container, text="清除日志",
                               command=self.clear_logs, style='File.TButton')
        clear_btn.pack(pady=(10, 0))

        return panel

    def select_onnx_file(self):
        """选择ONNX文件"""
        filename = filedialog.askopenfilename(
            title="选择ONNX模型文件",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        if filename:
            self.onnx_model_path.set(filename)
            self.log_message(f"已选择ONNX模型: {os.path.basename(filename)}", "success")

    def select_json_file(self):
        """选择JSON文件"""
        filename = filedialog.askopenfilename(
            title="选择网络结构JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.network_json_path.set(filename)
            self.log_message(f"已选择JSON文件: {os.path.basename(filename)}", "success")

    def select_op_library(self):
        """选择算子库目录"""
        dirname = filedialog.askdirectory(title="选择算子库目录 (Op_Library)")
        if dirname:
            self.op_library_path.set(dirname)
            self.log_message(f"已选择算子库: {os.path.basename(dirname)}", "success")

    def select_data_library(self):
        """选择数据库目录"""
        dirname = filedialog.askdirectory(title="选择数据库目录 (Data_Library)")
        if dirname:
            self.data_library_path.set(dirname)
            self.log_message(f"已选择数据库: {os.path.basename(dirname)}", "success")

    def log_message(self, message, level="info"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.update()

    def clear_logs(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)

    def update_stage_indicator(self, stage_id, status):
        """更新阶段指示器
        status: 'pending', 'running', 'completed', 'error'
        """
        if stage_id in self.stage_indicators:
            canvas, circle = self.stage_indicators[stage_id]

            colors = {
                'pending': '#4a4a5e',
                'running': '#3b82f6',
                'completed': '#10b981',
                'error': '#ef4444'
            }

            canvas.itemconfig(circle, fill=colors.get(status, '#4a4a5e'))
            canvas.update()

    def validate_config(self):
        """验证配置"""
        if not self.onnx_model_path.get():
            self.log_message("错误: 请选择ONNX模型文件", "error")
            return False
        if not self.network_json_path.get():
            self.log_message("错误: 请选择网络结构JSON文件", "error")
            return False
        if not self.op_library_path.get():
            self.log_message("错误: 请选择算子库目录", "error")
            return False
        if not self.data_library_path.get():
            self.log_message("错误: 请选择数据库目录", "error")
            return False
        return True

    def start_execution(self):
        """开始执行工具链"""
        if not self.validate_config():
            return

        if self.is_running:
            self.log_message("工具链正在运行中...", "warning")
            return

        # 在新线程中执行，避免阻塞UI
        thread = threading.Thread(target=self.run_pipeline)
        thread.daemon = True
        thread.start()

    def run_pipeline(self):
        """执行完整流程"""
        self.is_running = True
        self.execute_btn.config(state='disabled')
        self.current_stage.set("执行中...")

        # 重置所有指示器
        for stage_id in self.stage_indicators:
            self.update_stage_indicator(stage_id, 'pending')

        try:
            self.log_message("=" * 60, "info")
            self.log_message("神经网络硬件加速器工具链开始执行", "stage")
            self.log_message("=" * 60, "info")

            # 创建输出目录
            output_dir = self.output_dir.get()
            os.makedirs(output_dir, exist_ok=True)

            # 定义输出文件路径
            network_json = self.network_json_path.get()
            original_task = os.path.join(output_dir, "1_original_tasks.txt")
            aligned_task = os.path.join(output_dir, "1_aligned_tasks.txt")
            control_task = os.path.join(output_dir, "2_control_and_tasks.txt")
            task_addresses_json = os.path.join(output_dir, "task_addresses.json")
            full_config = os.path.join(output_dir, "3_full_config_with_data.txt")
            data_addresses_json = os.path.join(output_dir, "data_addresses.json")
            final_output = os.path.join(output_dir, "final_executable_config.txt")

            # 阶段0: ONNX转换（可选）
            if self.onnx_model_path.get():
                self.execute_stage0()

            # 阶段1: 生成任务指令
            self.execute_stage1(network_json, original_task, aligned_task)

            # 阶段2: 生成控制模块
            self.execute_stage2(aligned_task, control_task, network_json, task_addresses_json)

            # 阶段3: 链接数据模块
            self.execute_stage3(control_task, full_config, network_json, data_addresses_json)

            # 阶段4: 修改最终地址
            self.execute_stage4(full_config, final_output, task_addresses_json, data_addresses_json)

            self.log_message("=" * 60, "success")
            self.log_message("所有阶段执行完成！", "success")
            self.log_message(f"最终输出文件: {final_output}", "success")
            self.log_message("=" * 60, "success")

            self.current_stage.set("执行完成")

        except Exception as e:
            self.log_message(f"执行出错: {str(e)}", "error")
            self.current_stage.set("执行失败")
            import traceback
            self.log_message(traceback.format_exc(), "error")

        finally:
            self.is_running = False
            self.execute_btn.config(state='normal')

    def execute_stage0(self):
        """执行阶段0"""
        self.update_stage_indicator('stage0', 'running')
        self.current_stage.set("阶段0: ONNX模型解析")
        self.log_message("=" * 20 + " 阶段0: ONNX模型解析 " + "=" * 20, "stage")

        # 重定向标准输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            converter = stage0_onnx_to_json.ONNXToNetworkStructure(self.onnx_model_path.get())
            network_structure = converter.convert()
            converter.save_to_json(self.network_json_path.get())

            # 获取输出
            output = sys.stdout.getvalue()
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line, "info")

            self.update_stage_indicator('stage0', 'completed')
            self.log_message("阶段0完成", "success")

        except Exception as e:
            self.update_stage_indicator('stage0', 'error')
            raise
        finally:
            sys.stdout = old_stdout

    def execute_stage1(self, network_path, original_output, aligned_output):
        """执行阶段1"""
        self.update_stage_indicator('stage1', 'running')
        self.current_stage.set("阶段1: 任务指令生成")
        self.log_message("=" * 20 + " 阶段1: 任务指令生成 " + "=" * 20, "stage")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            stage1_task_generator.generate_task_instructions(
                network_path=network_path,
                library_path=self.op_library_path.get(),
                original_output=original_output,
                aligned_output=aligned_output
            )

            output = sys.stdout.getvalue()
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line, "info")

            self.update_stage_indicator('stage1', 'completed')
            self.log_message("阶段1完成", "success")

        except Exception as e:
            self.update_stage_indicator('stage1', 'error')
            raise
        finally:
            sys.stdout = old_stdout

    def execute_stage2(self, aligned_task, control_task, network_path, task_addresses):
        """执行阶段2"""
        self.update_stage_indicator('stage2', 'running')
        self.current_stage.set("阶段2: 控制信息配置")
        self.log_message("=" * 20 + " 阶段2: 控制信息配置 " + "=" * 20, "stage")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            stage2_control_generator.generate_control_module(
                aligned_task_file=aligned_task,
                control_task_output_file=control_task,
                network_path=network_path,
                task_address_output_file=task_addresses
            )

            output = sys.stdout.getvalue()
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line, "info")

            self.update_stage_indicator('stage2', 'completed')
            self.log_message("阶段2完成", "success")

        except Exception as e:
            self.update_stage_indicator('stage2', 'error')
            raise
        finally:
            sys.stdout = old_stdout

    def execute_stage3(self, control_task, full_output, network_path, data_addresses):
        """执行阶段3"""
        self.update_stage_indicator('stage3', 'running')
        self.current_stage.set("阶段3: 数据模块链接")
        self.log_message("=" * 20 + " 阶段3: 数据模块链接 " + "=" * 20, "stage")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            stage3_data_linker.link_data_module(
                control_task_file=control_task,
                full_output_file=full_output,
                network_path=network_path,
                db_root=self.data_library_path.get(),
                data_address_output_file=data_addresses
            )

            output = sys.stdout.getvalue()
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line, "info")

            self.update_stage_indicator('stage3', 'completed')
            self.log_message("阶段3完成", "success")

        except Exception as e:
            self.update_stage_indicator('stage3', 'error')
            raise
        finally:
            sys.stdout = old_stdout

    def execute_stage4(self, input_file, final_output, task_addresses, data_addresses):
        """执行阶段4"""
        self.update_stage_indicator('stage4', 'running')
        self.current_stage.set("阶段4: 地址修正")
        self.log_message("=" * 20 + " 阶段4: 地址修正 " + "=" * 20, "stage")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            stage4_address_modifier.modify_final_addresses(
                input_file=input_file,
                final_output_file=final_output,
                task_addresses_file=task_addresses,
                data_addresses_file=data_addresses
            )

            output = sys.stdout.getvalue()
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line, "info")

            self.update_stage_indicator('stage4', 'completed')
            self.log_message("阶段4完成", "success")

        except Exception as e:
            self.update_stage_indicator('stage4', 'error')
            raise
        finally:
            sys.stdout = old_stdout


def main():
    root = tk.Tk()
    app = ModernToolchainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()