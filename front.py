import sys
import os
import time
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QFileDialog, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class BackendThread(QThread):
    """后端执行线程，用于依次运行四个Python文件
    核心修改：将用户指定的输出路径传递给第4个脚本（4_MC_feature_map_initial_addr_modify.py）
    """
    # 定义信号：用于向前端发送输出信息和进度更新
    output_signal = pyqtSignal(str)  # 输出日志信号
    progress_signal = pyqtSignal(int)  # 进度条更新信号

    def __init__(self, model_path, operator_lib, db_path, output_path):
        super().__init__()
        # 接收前端传递的路径参数
        self.model_path = model_path  # 模型文件路径（network_structure.json）
        self.operator_lib = operator_lib  # 算子库目录
        self.db_path = db_path  # 数据库目录
        self.output_path = output_path  # 用户指定的最终输出文件路径（核心参数）
        self.running = True  # 线程运行状态标记

    def run(self):
        """依次执行四个Python文件，并捕获输出
        关键修改：给第4个脚本添加输出路径参数
        """
        # 定义需要执行的脚本列表，第4个脚本增加输出路径参数
        # 格式：(脚本文件名, [传递的参数列表])
        files_to_run = [
            ("1_test_taskModule.py", []),  # 第1个脚本：无参数
            ("2_test_taskModule_controlModule.py", []),  # 第2个脚本：无参数
            ("3_test_taskModule_controlModule_dataModule.py", []),  # 第3个脚本：无参数
            # 第4个脚本：传递用户指定的输出路径作为参数（核心修改点）
            ("4_MC_feature_map_initial_addr_modify.py", [self.output_path])
        ]

        # 保存原始工作目录（避免脚本执行路径影响）
        original_dir = os.getcwd()
        try:
            # 循环执行每个脚本
            for i, (file, args) in enumerate(files_to_run):
                if not self.running:  # 检查是否需要停止
                    break
                # 向前端发送当前执行的脚本信息
                self.output_signal.emit(f"\n==== 开始执行 {file} ====\n")
                # 更新进度条（4个脚本，每个占25%进度）
                self.progress_signal.emit((i + 1) * 25)

                # 构建执行命令：[Python解释器路径, 脚本文件, 参数1, 参数2...]
                cmd = [sys.executable, file] + args  # args为传递给脚本的参数列表

                # 设置环境变量（可在被调用的脚本中通过os.environ获取）
                env = os.environ.copy()
                env["MODEL_PATH"] = self.model_path  # 模型路径
                env["OPERATOR_LIB"] = self.operator_lib  # 算子库路径
                env["DB_PATH"] = self.db_path  # 数据库路径
                env["OUTPUT_PATH"] = self.output_path  # 输出路径（备用，供脚本读取）

                # 执行脚本并捕获输出
                process = subprocess.Popen(
                    cmd,  # 执行命令
                    cwd=original_dir,  # 保持原始工作目录
                    env=env,  # 传递环境变量
                    stdout=subprocess.PIPE,  # 捕获标准输出
                    stderr=subprocess.STDOUT,  # 标准错误也合并到stdout
                    text=True,  # 以文本模式处理输出
                    encoding="utf-8",  # 显式指定编码，避免中文乱码
                    bufsize=1  # 行缓冲，实时输出
                )

                # 实时读取脚本输出并发送到前端
                for line in process.stdout:
                    if not self.running:  # 如果用户点击停止，终止进程
                        process.terminate()
                        break
                    self.output_signal.emit(line)  # 发送输出到前端
                    time.sleep(0.005)  # 控制输出速度，避免界面卡顿

                # 等待进程结束并检查返回码
                process.wait()
                if process.returncode != 0:
                    self.output_signal.emit(f"执行 {file} 时出错，返回码: {process.returncode}\n")
                else:
                    self.output_signal.emit(f"==== {file} 执行完成 ====\n")

                # 脚本间添加短暂间隔，避免资源竞争
                time.sleep(0.05)

        except Exception as e:
            # 捕获执行过程中的异常并发送到前端
            self.output_signal.emit(f"执行过程中发生错误: {str(e)}\n")
        finally:
            # 执行完成后更新进度为100%
            self.progress_signal.emit(100)

    def stop(self):
        """停止线程执行（用户点击“停止”按钮时调用）"""
        if hasattr(self, 'backend_thread') and self.backend_thread.isRunning():
            self.running = False  # 设置停止标记
            self.backend_thread.wait()  # 等待线程结束
            self.output_text.append("操作已停止")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("操作已停止")


class NeuralNetworkConfigGUI(QMainWindow):
    """神经网络配置生成工具主界面"""
    def __init__(self):
        super().__init__()
        self.initUI()  # 初始化界面

    def initUI(self):
        """初始化用户界面元素"""
        self.setWindowTitle("神经网络配置生成工具")  # 窗口标题
        self.setGeometry(100, 100, 900, 700)  # 窗口位置和大小

        # 中央部件（用于承载所有界面元素）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局（垂直布局）
        main_layout = QVBoxLayout(central_widget)

        # conv_12x12x20_8x8x10_k5_s1_p0. 路径配置区域
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)  # 控件间距
        path_layout.addWidget(QLabel("路径配置", styleSheet="font-weight: bold; font-size: 14pt;"))

        # conv_12x12x20_8x8x10_k5_s1_p0.conv_12x12x20_8x8x10_k5_s1_p0 模型文件路径（network_structure.json）
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型文件路径:"))
        self.model_path_edit = QLineEdit("network_structure.json")  # 默认路径
        model_layout.addWidget(self.model_path_edit)
        model_btn = QPushButton("浏览")
        model_btn.clicked.connect(self.browse_model_path)  # 绑定浏览按钮事件
        model_layout.addWidget(model_btn)
        path_layout.addLayout(model_layout)

        # conv_12x12x20_8x8x10_k5_s1_p0.conv_8x8x20_8x8x10_k3_s1_p1 算子库目录
        operator_layout = QHBoxLayout()
        operator_layout.addWidget(QLabel("算子库目录:"))
        self.operator_lib_edit = QLineEdit("Op_Library")  # 默认路径
        operator_layout.addWidget(self.operator_lib_edit)
        operator_btn = QPushButton("浏览")
        operator_btn.clicked.connect(self.browse_operator_lib)  # 绑定浏览按钮事件
        operator_layout.addWidget(operator_btn)
        path_layout.addLayout(operator_layout)

        # conv_12x12x20_8x8x10_k5_s1_p0.conv_8x8x20_8x8x10_k3_s1_p1 数据库目录
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("数据库目录:"))
        self.db_path_edit = QLineEdit("Data_Library")  # 默认路径
        db_layout.addWidget(self.db_path_edit)
        db_btn = QPushButton("浏览")
        db_btn.clicked.connect(self.browse_db_path)  # 绑定浏览按钮事件
        db_layout.addWidget(db_btn)
        path_layout.addLayout(db_layout)

        # conv_12x12x20_8x8x10_k5_s1_p0.conv_16x16x20_12x12x10_k5_s1_p0 最终输出文件路径（核心输入框）
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("最终输出文件路径:"))
        # 默认路径，用户可修改
        self.output_path_edit = QLineEdit("./控制信息配置+总任务指令配置+总数据信息配置_new.txt")
        output_layout.addWidget(self.output_path_edit)
        output_btn = QPushButton("浏览")
        output_btn.clicked.connect(self.browse_output_path)  # 绑定浏览按钮事件
        output_layout.addWidget(output_btn)
        path_layout.addLayout(output_layout)

        # 将路径配置区域添加到主布局
        main_layout.addLayout(path_layout)

        # 分割线（视觉分隔）
        line = QLabel("-" * 80)
        line.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(line)

        # conv_8x8x20_8x8x10_k3_s1_p1. 执行控制区域（开始/停止按钮+进度条）
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始生成")
        self.start_btn.clicked.connect(self.start_backend)  # 绑定开始按钮事件
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("font-size: 12pt;")

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_backend)  # 绑定停止按钮事件
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("font-size: 12pt;")
        self.stop_btn.setEnabled(False)  # 初始状态禁用

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)  # 初始进度0%

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.progress_bar)
        main_layout.addLayout(control_layout)

        # conv_8x8x20_8x8x10_k3_s1_p1. 后端输出显示区域
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("后端链接过程", styleSheet="font-weight: bold; font-size: 14pt;"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)  # 只读，防止用户编辑
        self.output_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")  # 等宽字体，适合显示日志
        output_layout.addWidget(self.output_text)
        main_layout.addLayout(output_layout)

        # 状态栏（显示当前状态）
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    # 以下为路径选择相关的事件处理函数
    def browse_model_path(self):
        """浏览选择模型文件路径（.json）"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "JSON Files (*.json)")
        if file_path:  # 如果用户选择了文件
            self.model_path_edit.setText(file_path)

    def browse_operator_lib(self):
        """浏览选择算子库目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择算子库目录")
        if dir_path:  # 如果用户选择了目录
            self.operator_lib_edit.setText(dir_path)

    def browse_db_path(self):
        """浏览选择数据库目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据库目录")
        if dir_path:  # 如果用户选择了目录
            self.db_path_edit.setText(dir_path)

    def browse_output_path(self):
        """浏览选择最终输出文件路径（.txt）"""
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出文件", "", "Text Files (*.txt)")
        if file_path:  # 如果用户选择了路径
            self.output_path_edit.setText(file_path)

    def start_backend(self):
        """启动后端执行线程（用户点击“开始生成”时调用）"""
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 清空输出区域
        self.output_text.clear()
        # 重置进度条
        self.progress_bar.setValue(0)
        # 更新状态栏
        self.status_bar.showMessage("正在生成配置...")

        # 获取用户输入的所有路径
        model_path = self.model_path_edit.text()
        operator_lib = self.operator_lib_edit.text()
        db_path = self.db_path_edit.text()
        output_path = self.output_path_edit.text()  # 核心：用户指定的输出路径

        # 简单校验路径是否存在（避免明显错误）
        if not os.path.exists(model_path):
            self.output_text.append(f"错误: 模型文件 {model_path} 不存在")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("错误: 模型文件不存在")
            return
        if not os.path.exists(operator_lib):
            self.output_text.append(f"错误: 算子库目录 {operator_lib} 不存在")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("错误: 算子库目录不存在")
            return
        if not os.path.exists(db_path):
            self.output_text.append(f"错误: 数据库目录 {db_path} 不存在")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("错误: 数据库目录不存在")
            return

        # 创建后端线程并传递参数（包含用户指定的输出路径）
        self.backend_thread = BackendThread(model_path, operator_lib, db_path, output_path)
        # 绑定线程信号：输出日志和进度更新
        self.backend_thread.output_signal.connect(self.append_output)
        self.backend_thread.progress_signal.connect(self.update_progress)
        # 线程结束后更新界面状态
        self.backend_thread.finished.connect(self.backend_finished)
        # 启动线程
        self.backend_thread.start()

    def stop_backend(self):
        """停止后端执行线程（用户点击“停止”时调用）"""
        if hasattr(self, 'backend_thread') and self.backend_thread.isRunning():
            self.backend_thread.stop()  # 调用线程的停止方法
            self.output_text.append("操作已停止")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("操作已停止")

    def append_output(self, text):
        """向前端输出区域添加内容"""
        self.output_text.append(text)
        # 自动滚动到底部，方便查看最新输出
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def backend_finished(self):
        """后端线程执行完成后调用"""
        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # 更新状态栏
        self.status_bar.showMessage("配置生成完成")
        # 提示用户执行完成
        self.output_text.append("\n==== 所有操作已完成 ====")


if __name__ == "__main__":
    # 启动应用
    app = QApplication(sys.argv)
    gui = NeuralNetworkConfigGUI()
    gui.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入事件循环