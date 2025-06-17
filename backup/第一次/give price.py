import pandas as pd
import numpy as np
from main import RTBBiddingSystem

def bid_day1(model_system, test_data_path, student_id, day_budget=5000):
    """
    Day1 出價邏輯
    
    Args:
        model_system: 已訓練好的 RTBBiddingSystem 實例
        test_data_path: 測試資料路徑
        student_id: 學生ID
        day_budget: 日預算
    """
    
    # 載入測試資料
    test_data = pd.read_csv(test_data_path)
    
    # 初始化預算控制
    remaining_budget = day_budget
    hourly_budget = [day_budget // 24] * 24  # 平均分配到24小時
    spent_hour = [0] * 24
    
    # 出價參數
    pctr_min = 1e-4
    rho_cut = 2e-5
    
    bid_results = []
    
    print(f"開始 Day1 出價，總預算: {day_budget}")
    
    for idx, row in test_data.iterrows():
        # 提取時間特徵
        if 'timestamp' in row:
            hour = pd.to_datetime(row['timestamp']).hour
        else:
            hour = 0
        
        bid_id = row.get('bid_id', idx)
        
        # 檢查時段預算
        if spent_hour[hour] >= hourly_budget[hour]:
            bid_price = 0
        else:
            # 預測點擊率
            pctr = model_system.predict_CTR(row)
            
            if pctr < pctr_min:
                bid_price = 0
            else:
                # 預測勝價
                win_price = model_system.predict_winprice(row)
                
                if win_price <= 0:
                    bid_price = 0
                else:
                    # 計算性價比
                    rho = pctr / win_price
                    
                    if rho < rho_cut or remaining_budget < win_price + 1:
                        bid_price = 0
                    else:
                        bid_price = win_price + 1
                        remaining_budget -= bid_price
                        spent_hour[hour] += bid_price
        
        bid_results.append({
            'bid_id': bid_id,
            'bid_price': bid_price
        })
        
        # 進度顯示
        if idx % 5000 == 0:
            print(f"處理進度: {idx}/{len(test_data)}, 剩餘預算: {remaining_budget}")
    
    # 儲存結果
    result_df = pd.DataFrame(bid_results)
    output_file = f"{student_id}_day1.csv"
    result_df.to_csv(output_file, index=False)
    
    # 統計資訊
    total_bids = (result_df['bid_price'] > 0).sum()
    total_spent = result_df['bid_price'].sum()
    
    print(f"\nDay1 出價完成:")
    print(f"輸出檔案: {output_file}")
    print(f"總出價次數: {total_bids}")
    print(f"總支出: {total_spent}")
    print(f"預算使用率: {total_spent/day_budget*100:.2f}%")
    
    return result_df

if __name__ == "__main__":
    # 載入已訓練的模型
    rtb_system = RTBBiddingSystem(student_id="your_student_id")
    rtb_system.load_data()
    rtb_system.prepare_training_data()
    rtb_system.train_ctr_model()
    
    # 訓練 Win-Price 模型
    rtb_system.train_winprice_model()
    
    # 執行出價
    bid_day1(rtb_system, 'data/test_day1.csv', 'your_student_id')