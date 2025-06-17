import pandas as pd

def analyze_bid_vs_actual():
    """
    比較出價與實際得標價的差異
    """
    # 讀取檔案
    bid_file = r'C:\Users\ian32\Downloads\final\上傳版本\M36134016_day1.csv'
    result_file = r'C:\Users\ian32\Downloads\final\M36134016_day1_rst.csv'
    
    df_bid = pd.read_csv(bid_file)
    df_result = pd.read_csv(result_file)
    
    print("=== 出價 vs 實際得標價分析 ===\n")
    
    # 合併資料 - 找出相同 bid_id 的記錄
    merged = pd.merge(
        df_bid, 
        df_result, 
        on='bid_id', 
        suffixes=('_my_bid', '_actual_win'),
        how='inner'
    )
    
    if len(merged) == 0:
        print("❌ 沒有找到相同的 bid_id，無法比較")
        return
    
    print(f"📊 找到 {len(merged)} 筆可比較的記錄\n")
    
    # 基本統計
    print("1. 基本統計:")
    print(f"   我的出價 - 平均: {merged['paying_price_my_bid'].mean():.2f}, "
          f"中位數: {merged['paying_price_my_bid'].median():.2f}, "
          f"範圍: {merged['paying_price_my_bid'].min()}-{merged['paying_price_my_bid'].max()}")
    
    print(f"   實際得標價 - 平均: {merged['paying_price_actual_win'].mean():.2f}, "
          f"中位數: {merged['paying_price_actual_win'].median():.2f}, "
          f"範圍: {merged['paying_price_actual_win'].min()}-{merged['paying_price_actual_win'].max()}")
    
    # 計算差異
    merged['price_diff'] = merged['paying_price_my_bid'] - merged['paying_price_actual_win']
    merged['price_ratio'] = merged['paying_price_my_bid'] / merged['paying_price_actual_win']
    
    print(f"\n2. 出價差異分析:")
    print(f"   平均差異: {merged['price_diff'].mean():.2f}")
    print(f"   差異標準差: {merged['price_diff'].std():.2f}")
    print(f"   出價/得標價比率 - 平均: {merged['price_ratio'].mean():.2f}")
    
    # 分類分析
    over_bid = (merged['paying_price_my_bid'] > merged['paying_price_actual_win']).sum()
    under_bid = (merged['paying_price_my_bid'] < merged['paying_price_actual_win']).sum()
    exact_bid = (merged['paying_price_my_bid'] == merged['paying_price_actual_win']).sum()
    
    print(f"\n3. 出價策略分析:")
    print(f"   出價過高: {over_bid} 筆 ({over_bid/len(merged)*100:.1f}%)")
    print(f"   出價過低: {under_bid} 筆 ({under_bid/len(merged)*100:.1f}%)")
    print(f"   出價剛好: {exact_bid} 筆 ({exact_bid/len(merged)*100:.1f}%)")
    
    # 詳細記錄
    print(f"\n4. 詳細比較 (前20筆):")
    print("-" * 80)
    comparison_df = merged[['bid_id', 'paying_price_my_bid', 'paying_price_actual_win', 'price_diff', 'click']].head(20)
    comparison_df.columns = ['bid_id', '我的出價', '實際得標價', '差異', '是否點擊']
    print(comparison_df.to_string(index=False))
    
    # 出價效率分析
    print(f"\n5. 出價效率分析:")
    
    # 按差異範圍分組
    bins = [-float('inf'), -10, -5, 0, 5, 10, float('inf')]
    labels = ['低估>10', '低估5-10', '低估<5', '高估<5', '高估5-10', '高估>10']
    merged['diff_category'] = pd.cut(merged['price_diff'], bins=bins, labels=labels)
    
    diff_analysis = merged.groupby('diff_category').agg({
        'bid_id': 'count',
        'click': 'sum'
    }).rename(columns={'bid_id': '出價次數', 'click': '點擊次數'})
    
    diff_analysis['點擊率%'] = (diff_analysis['點擊次數'] / diff_analysis['出價次數'] * 100).round(2)
    print(diff_analysis)
    
    # 過度出價的損失
    overpay_loss = merged[merged['price_diff'] > 0]['price_diff'].sum()
    print(f"\n6. 過度出價損失: {overpay_loss:.0f} 元")
    
    # 可能錯失的機會
    underbid_count = (merged['price_diff'] < -5).sum()
    print(f"   可能因出價太低錯失的機會: {underbid_count} 次")
    
    # 儲存詳細比較結果
    output_file = r'C:\Users\ian32\Downloads\final\bid_analysis.csv'
    merged.to_csv(output_file, index=False)
    print(f"\n💾 詳細分析結果已儲存至: {output_file}")
    
    return merged

# 執行分析
if __name__ == "__main__":
    analysis_result = analyze_bid_vs_actual()