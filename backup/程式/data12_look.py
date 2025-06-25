import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from lifelines import WeibullAFTFitter
from collections import Counter
from sklearn.metrics import classification_report, f1_score

class RTBBiddingSystem:
    def __init__(self, student_id="M36134016"):
        self.student_id = student_id
        self.DAY_BUDGET = 5000
        self.pctr_min = 1e-4  # 最低可接受的預測點擊率
        self.rho_cut = 2e-5   # 性價比門檻 (pCTR / win_price)

        # 生成統一的時間戳
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 動態調整每小時預算，根據歷史競價情況分配，而不是均分
        hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.3, 1.1, 1.0, 
                        1.0, 1.1, 1.3, 1.5, 1.8, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4]
        total_weight = sum(hour_weights)
        self.hourly_budget = [(self.DAY_BUDGET * w / total_weight) for w in hour_weights]

        # 模型 - 擴展為三個模型
        self.ctr_model_lgb = None      # LightGBM 模型
        self.ctr_model_deepfm = None   # DeepFM 模型  
        self.ctr_model_xdeepfm = None  # xDeepFM 模型
        self.winprice_model = None
        self.feature_encoders = {}  # 儲存 LabelEncoders 和其他轉換器
        self.processed_train_df = None # 儲存處理過的訓練資料以供重用

        # DeepCTR 相關
        self.deepctr_feature_columns = None
        self.deepctr_feature_names = None
        self.deepctr_train_processed = None
        self.deepctr_scalers = {}

        # 資料
        self.train_data = None
        self.test_day2 = None
        self.X_train = None
        self.y_ctr = None
        self.feature_columns = None # 用於模型訓練的特徵欄位名稱
        self.winprice_features = None # 用於 Win-Price 模型訓練的特徵

        # 模型選擇 (預設使用 LightGBM)
        self.active_ctr_model = 'lgb'  # 'lgb', 'deepfm', 'xdeepfm'

    def load_data(self):
        """載入訓練資料與測試資料"""
        print("載入資料中...")
        try:
            self.train_data = pd.read_csv('data/train.csv')
            self.test_day2 = pd.read_csv('data/test_day2.csv')
            print(f"訓練集大小: {self.train_data.shape}")
            print(f"測試集大小: {self.test_day2.shape}")
        except FileNotFoundError as e:
            print(f"錯誤：找不到資料檔案 {e.filename}")
            raise
        # 嘗試載入特徵說明檔案 (可選)
        try:
            self.feature_meaning = pd.read_excel('data/Feature_Meaning.xlsx')
        except FileNotFoundError:
            print("提示：未找到特徵說明檔案 'data/Feature_Meaning.xlsx'")
        except Exception as e:
            print(f"讀取特徵說明檔案時發生錯誤: {e}")

    def preprocess_features(self, df, is_train=True):
        """特徵工程"""
        df_processed = df.copy()

        # 1. 時間特徵
        if 'timestamp' in df_processed.columns:
            try:
                df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], format='ISO8601', errors='coerce')
                df_processed['hour'] = df_processed['timestamp'].dt.hour
                df_processed['weekday'] = df_processed['timestamp'].dt.weekday
                # 處理轉換失敗的 NaT (如果有的話)
                if df_processed['hour'].isnull().any():
                    print("警告：'hour' 特徵中存在 NaNs，可能由 'timestamp' 轉換失敗導致。將用中位數填充。")
                    df_processed['hour'].fillna(df_processed['hour'].median(), inplace=True)
                if df_processed['weekday'].isnull().any():
                    print("警告：'weekday' 特徵中存在 NaNs，將用中位數填充。")
                    df_processed['weekday'].fillna(df_processed['weekday'].median(), inplace=True)
            except Exception as e:
                print(f"處理 'timestamp' 時發生錯誤: {e}. 'hour' 和 'weekday' 可能未正確產生。")
                # 提供預設值或更穩健的錯誤處理
                if 'hour' not in df_processed.columns: df_processed['hour'] = 0
                if 'weekday' not in df_processed.columns: df_processed['weekday'] = 0


        # 2. 數值特徵缺失值填補
        numeric_features = ['ad_slot_width', 'ad_slot_height', 'ad_slot_floor_price']
        # 'bidding_price' 通常在訓練集中都有，測試集中可能沒有，這裡不主動填充測試集的 bidding_price
        # 如果 bidding_price 在訓練集也可能缺失，則應加入
        if is_train and 'bidding_price' in df_processed.columns:
             numeric_features.append('bidding_price')

        for col in numeric_features:
            if col in df_processed.columns:
                median_val = None
                if is_train:
                    median_val = df_processed[col].median()
                    self.feature_encoders[f'{col}_median'] = median_val
                else:
                    median_val = self.feature_encoders.get(f'{col}_median', df_processed[col].median()) # 測試集也用中位數以防萬一

                if pd.isna(median_val) and is_train: # 如果訓練集的中位數是 NaN (例如全為 NaN 的欄)
                    print(f"警告：訓練集中特徵 '{col}' 的中位數為 NaN。將用 0 填充。")
                    median_val = 0
                    self.feature_encoders[f'{col}_median'] = median_val
                elif pd.isna(median_val) and not is_train: # 測試集的中位數是 NaN
                     median_val = self.feature_encoders.get(f'{col}_median', 0) # 從訓練集取，若無則用0

                df_processed[col] = df_processed[col].fillna(median_val)

        # 3. 類別特徵處理 (高頻截斷 + LabelEncoding)
        categorical_features = ['domain', 'url', 'city', 'region', 'ad_exchange',
                                'user_agent', 'ad_slot_id', 'creative_id']
        for col in categorical_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str) # 確保為字串類型
                if is_train:
                    value_counts = df_processed[col].value_counts()
                    top_values = value_counts.head(50).index.tolist()
                    self.feature_encoders[f'{col}_top_values'] = top_values
                    df_processed[col] = df_processed[col].apply(lambda x: x if x in top_values else 'other')
                    
                    le = LabelEncoder()
                    # 確保 'other' 類別被學習，即使它在截斷後不存在於樣本中
                    all_possible_values = df_processed[col].unique().tolist()
                    if 'other' not in all_possible_values:
                        all_possible_values.append('other')
                    le.fit(all_possible_values)
                    
                    df_processed[col] = le.transform(df_processed[col])
                    self.feature_encoders[f'{col}_encoder'] = le
                else:
                    top_values = self.feature_encoders.get(f'{col}_top_values', [])
                    le = self.feature_encoders.get(f'{col}_encoder')
                    if le is not None:
                        df_processed[col] = df_processed[col].apply(lambda x: x if x in top_values else 'other')
                        # 處理測試集中的新類別 (LabelEncoder 不認識的)
                        df_processed[col] = df_processed[col].apply(lambda x: x if x in le.classes_ else 'other')
                        df_processed[col] = le.transform(df_processed[col])
                    else: # 如果沒有 encoder，可能表示此特徵在訓練時不存在或未處理
                        print(f"警告：測試集中特徵 '{col}' 缺少編碼器，將填充為 0。")
                        df_processed[col] = 0


        # 4. IP 特徵處理
        if 'ip' in df_processed.columns:
            df_processed['ip_prefix'] = df_processed['ip'].astype(str).apply(
                lambda x: '.'.join(x.split('.')[:2]) if '.' in x else 'unknown_ip'
            )
            if is_train:
                # 強制加入 'unknown_ip' 進 LabelEncoder
                all_ip_prefixes = df_processed['ip_prefix'].unique().tolist()
                if 'unknown_ip' not in all_ip_prefixes:
                    all_ip_prefixes.append('unknown_ip')
                le_ip = LabelEncoder()
                le_ip.fit(all_ip_prefixes)
                df_processed['ip_prefix'] = le_ip.transform(df_processed['ip_prefix'])
                self.feature_encoders['ip_encoder'] = le_ip
            else:
                le_ip = self.feature_encoders.get('ip_encoder')
                if le_ip is not None:
                    df_processed['ip_prefix'] = df_processed['ip_prefix'].apply(
                        lambda x: x if x in le_ip.classes_ else 'unknown_ip'
                    )
                    df_processed['ip_prefix'] = le_ip.transform(df_processed['ip_prefix'])
                else:
                    print("警告：測試集中 IP 特徵缺少編碼器，將填充為 0。")
                    df_processed['ip_prefix'] = 0


        # 5. user_tags 處理 (熱門標籤二元化)
        if 'user_tags' in df_processed.columns:
            df_processed['user_tags'] = df_processed['user_tags'].fillna('')
            if is_train:
                all_tags = []
                for tags_list in df_processed['user_tags'].apply(lambda x: x.split(',') if x else []):
                    all_tags.extend(tags_list)
                tag_counts = Counter(all_tags)
                # 選擇前20個最常見的標籤，排除空字串標籤 (如果有的話)
                top_tags = [tag for tag, count in tag_counts.most_common(25) if tag][:20]
                self.feature_encoders['top_tags'] = top_tags

            top_tags = self.feature_encoders.get('top_tags', [])
            for tag in top_tags:
                df_processed[f'tag_{tag}'] = df_processed['user_tags'].apply(lambda x: 1 if tag in x.split(',') else 0)

        # 6. key_page_url 特徵處理
        if 'key_page_url' in df_processed.columns:
            df_processed['key_page_url'] = df_processed['key_page_url'].astype(str)
            if is_train:
                value_counts = df_processed['key_page_url'].value_counts()
                top_urls = value_counts.head(50).index.tolist()
                self.feature_encoders['key_page_url_top'] = top_urls
                df_processed['key_page_url_processed'] = df_processed['key_page_url'].apply(lambda x: x if x in top_urls else 'other_url')
                
                le_url = LabelEncoder()
                all_possible_urls = df_processed['key_page_url_processed'].unique().tolist()
                if 'other_url' not in all_possible_urls:
                     all_possible_urls.append('other_url')
                le_url.fit(all_possible_urls)

                df_processed['key_page_url_processed'] = le_url.transform(df_processed['key_page_url_processed'])
                self.feature_encoders['key_page_url_encoder'] = le_url
            else:
                top_urls = self.feature_encoders.get('key_page_url_top', [])
                le_url = self.feature_encoders.get('key_page_url_encoder')
                if le_url is not None:
                    df_processed['key_page_url_processed'] = df_processed['key_page_url'].apply(lambda x: x if x in top_urls else 'other_url')
                    df_processed['key_page_url_processed'] = df_processed['key_page_url_processed'].apply(lambda x: x if x in le_url.classes_ else 'other_url')
                    df_processed['key_page_url_processed'] = le_url.transform(df_processed['key_page_url_processed'])
                else:
                    print("警告：測試集中 key_page_url 特徵缺少編碼器，將填充為 0。")
                    df_processed['key_page_url_processed'] = 0
            # 移除原始 key_page_url，如果它不是數值型
            if 'key_page_url' in df_processed.columns and df_processed['key_page_url'].dtype == 'object':
                 df_processed.drop('key_page_url', axis=1, inplace=True, errors='ignore')


        # 新特徵工程部分
        # 1. 時間特徵增強
        df_processed['time_segment'] = pd.cut(
            df_processed['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        # 將時段轉換為數值
        if is_train:
            le_time = LabelEncoder()
            df_processed['time_segment'] = le_time.fit_transform(df_processed['time_segment'])
            self.feature_encoders['time_segment_encoder'] = le_time
        else:
            le_time = self.feature_encoders.get('time_segment_encoder')
            if le_time is not None:
                df_processed['time_segment'] = le_time.transform(df_processed['time_segment'])
            else:
                df_processed['time_segment'] = 0

        # 2. 添加交互特徵 - domain與時段的組合
        if 'domain' in df_processed.columns and 'time_segment' in df_processed.columns:
            df_processed['domain_time'] = df_processed['domain'] * 10 + df_processed['time_segment']


        # 確保所有 winprice_features 都有值
        if self.winprice_features is None:
            self.winprice_features = []  # 初始化為空列表，避免錯誤

        for col in self.winprice_features:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(0)

        return df_processed


# 新增主程式區塊
if __name__ == "__main__":
    # 初始化系統
    rtb_system = RTBBiddingSystem()
    rtb_system.load_data()

    # 預處理訓練資料
    df_processed = rtb_system.preprocess_features(rtb_system.train_data, is_train=True)

    print("處理後的欄位：")
    print(df_processed.columns.tolist())

    print("\n前 5 筆資料：")
    print(df_processed.head())