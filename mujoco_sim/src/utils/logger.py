import csv
import os
from datetime import datetime

class DataLogger:
    def __init__(self, log_dir="./logs", prefix="hydro_debug"):
        # 1. 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 2. 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{log_dir}/{prefix}_{timestamp}.csv"
        
        # 3. 初始化文件和表头
        self.csv_file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        self.header = [
            "Time", 
            "u", "v", "w", "p", "q", "r",                  # Body 速度
            "Fx_H", "Fy_H", "Fz_H", "Tx_H", "Ty_H", "Tz_H" # 水动力(Body系)
        ]
        self.writer.writerow(self.header)
        print(f"日志记录已启动: {self.filename}")

    def log(self, time, velocity, forces):
        """
        time: 当前仿真时间
        velocity: 6维速度向量
        forces: 6维力/力矩向量
        """
        row = [time] + list(velocity) + list(forces)
        self.writer.writerow(row)

    def close(self):
        self.csv_file.close()
        print(f"日志已成功保存至: {self.filename}")