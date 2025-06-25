import pandas as pd
from user_agents import parse
from tqdm import tqdm

# 讀取資料
df = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')

# 建立空欄位
df['browser_family'] = ''
df['os_family'] = ''
df['device_family'] = ''

# 解析 user_agent 欄位，處理 NaN
for idx, ua_string in tqdm(df['user_agent'].items(), total=len(df)):
    if pd.isna(ua_string):
        ua = parse("")
    else:
        ua = parse(ua_string)
    df.at[idx, 'browser_family'] = ua.browser.family
    df.at[idx, 'os_family'] = ua.os.family
    df.at[idx, 'device_family'] = ua.device.family

# 解析 user_agent 欄位
for idx, ua_string in tqdm(df['user_agent'].items(), total=len(df)):
    ua = parse(ua_string)
    df.at[idx, 'browser_family'] = ua.browser.family
    df.at[idx, 'os_family'] = ua.os.family
    df.at[idx, 'device_family'] = ua.device.family

# 分析瀏覽器與 click 的關係
browser_summary = df.groupby('browser_family')['click'].agg(['sum', 'count', 'mean']).rename(
    columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})
print("瀏覽器與 click 的關係：")
print(browser_summary.sort_values('click_sum', ascending=False))

# 分析作業系統與 click 的關係
os_summary = df.groupby('os_family')['click'].agg(['sum', 'count', 'mean']).rename(
    columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})
print("\n作業系統與 click 的關係：")
print(os_summary.sort_values('click_sum', ascending=False))

# 分析裝置與 click 的關係
device_summary = df.groupby('device_family')['click'].agg(['sum', 'count', 'mean']).rename(
    columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})
print("\n裝置與 click 的關係：")
print(device_summary.sort_values('click_sum', ascending=False))


#              click_sum   total  click_rate
# ad_exchange
# 1                  410  573025    0.000716
# 2                  388  522498    0.000743
# 3                  521  664786    0.000784
# import pandas as pd

# # 讀取資料
# df = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')

# # 分析 ad_exchange 與 click 的關係
# ad_exchange_summary = df.groupby('ad_exchange')['click'].agg(['sum', 'count', 'mean'])
# ad_exchange_summary = ad_exchange_summary.rename(columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})

# print(ad_exchange_summary)

# # 判斷哪個 ad_exchange 的 click 總數較高
# higher_click_exchange = ad_exchange_summary['click_sum'].idxmax()
# print(f"\nad_exchange = {higher_click_exchange} 的 click 總數較高 ({ad_exchange_summary.loc[higher_click_exchange, 'click_sum']})")



#                     click_sum   total  click_rate
# ad_slot_visibility
# 0                         666  864550    0.000770
# 1                         197  183762    0.001072
# 2                         452  703587    0.000642
# 255                         4    8410    0.000476

# import pandas as pd

# # 讀取資料
# df = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')

# # 分析 ad_slot_visibility 與 click 的關係
# ad_slot_vis_summary = df.groupby('ad_slot_visibility')['click'].agg(['sum', 'count', 'mean'])
# ad_slot_vis_summary = ad_slot_vis_summary.rename(columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})

# print(ad_slot_vis_summary)

# # 判斷哪個 ad_slot_visibility 的 click 總數較高
# higher_click_vis = ad_slot_vis_summary['click_sum'].idxmax()
# print(f"\nad_slot_visibility = {higher_click_vis} 的 click 總數較高 ({ad_slot_vis_summary.loc[higher_click_vis, 'click_sum']})")


# import pandas as pd

#                 click_sum    total  click_rate
# ad_slot_format
# 0                     909  1187284    0.000766
# 1                     410   573025    0.000716


# # 讀取資料
# df = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')

# # 分析 ad_slot_format 與 click 的關係
# ad_slot_summary = df.groupby('ad_slot_format')['click'].agg(['sum', 'count', 'mean'])
# ad_slot_summary = ad_slot_summary.rename(columns={'sum': 'click_sum', 'count': 'total', 'mean': 'click_rate'})

# print(ad_slot_summary)

# # 判斷哪個 ad_slot_format 的 click 總數較高
# higher_click_format = ad_slot_summary['click_sum'].idxmax()
# print(f"\nad_slot_format = {higher_click_format} 的 click 總數較高 ({ad_slot_summary.loc[higher_click_format, 'click_sum']})")


# import pandas as pd
# import matplotlib.pyplot as plt

# # 讀取資料
# df = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')

# # 依據 IP 分組，計算出現次數與點擊數，並取前 20 名
# df_ip_clicks = (
#     df.groupby('ip')
#       .agg(frequency=('ip', 'count'), click=('click', 'sum'))
#       .nlargest(100, 'frequency')
# )

# # 畫出點擊數長條圖並存檔
# ax = df_ip_clicks['click'].plot(kind='bar', color='skyblue', title='Top 100 IPs by Frequency and Clicks')
# plt.ylabel('Click Count')
# plt.xlabel('IP')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(r'C:\Users\ian32\Downloads\final\top100_ip_clicks.png', dpi=300)