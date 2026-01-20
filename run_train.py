#!/usr/bin/env python3
from neupan import neupan
import os

if __name__ == '__main__':
    print(">>> 正在启动训练...")
    
    # 确保在正确的目录
    os.chdir('/home/lyb/neupan_ws/src/NeuPAN')
    
    # 初始化并开始训练
    neupan_planner = neupan.init_from_yaml('configs/train_scout_mini.yaml')
    neupan_planner.train_dune()