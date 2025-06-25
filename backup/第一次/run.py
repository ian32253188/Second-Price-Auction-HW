"""
RTB 競價系統執行腳本
使用方法: python run.py
"""

import os
import sys
from backup.程式.main import RTBBiddingSystem

def check_data_files():
    """檢查必要的資料檔案是否存在"""
    required_files = [
        'data/train.csv',
        'data/test_day1.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下必要檔案:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def main():
    print("=== RTB 競價系統 ===")
    
    # 檢查資料檔案
    if not check_data_files():
        print("請確保資料檔案存在後再執行")
        return
    
    # 設定學生ID（請修改為您的學號）
    student_id = "M36134016"
    if not student_id:
        student_id = "default_student_id"
    
    try:
        # 初始化系統
        print("\n1. 初始化系統...")
        rtb_system = RTBBiddingSystem(student_id=student_id)
        
        # 載入資料
        print("\n2. 載入資料...")
        rtb_system.load_data()
        
        # 特徵工程
        print("\n3. 特徵工程...")
        rtb_system.prepare_training_data()
        # 檢查 click 標籤分布
        print("click 標籤分布：")
        print(rtb_system.y_ctr.value_counts())
        
        # 訓練模型
        print("\n4. 訓練 CTR 模型...")
        rtb_system.train_ctr_model()
        
        print("\n5. 訓練 Win-Price 模型...")
        rtb_system.train_winprice_model()
        
        # Day1 出價
        print("\n6. 執行 Day1 出價...")
        result = rtb_system.bid_day1()
        
        print(f"\n=== 執行完成 ===")
        print(f"結果已儲存至: {student_id}_day1.csv")
        
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()