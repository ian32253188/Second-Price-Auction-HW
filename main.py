import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lifelines import WeibullAFTFitter
import warnings
import os
from collections import Counter

warnings.filterwarnings('ignore')

class RTBBiddingSystem:
    def __init__(self, student_id="M36134016"):
        self.student_id = student_id
        self.DAY_BUDGET = 5000
        self.pctr_min = 1e-4  # 最低可接受的預測點擊率
        self.rho_cut = 2e-5   # 性價比門檻 (pCTR / win_price)
        self.hourly_budget = [self.DAY_BUDGET // 24] * 24 # 平均每小時預算

        # 模型
        self.ctr_model = None
        self.winprice_model = None
        self.feature_encoders = {}  # 儲存 LabelEncoders 和其他轉換器
        self.processed_train_df = None # 儲存處理過的訓練資料以供重用

        # 資料
        self.train_data = None
        self.test_day1 = None
        self.X_train = None
        self.y_ctr = None
        self.feature_columns = None # 用於模型訓練的特徵欄位名稱
        self.winprice_features = None # 用於 Win-Price 模型訓練的特徵

    def load_data(self):
        """載入訓練資料與測試資料"""
        print("載入資料中...")
        try:
            self.train_data = pd.read_csv('data/train.csv')
            self.test_day1 = pd.read_csv('data/test_day1.csv')
            print(f"訓練集大小: {self.train_data.shape}")
            print(f"測試集大小: {self.test_day1.shape}")
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


        return df_processed

    def prepare_training_data(self):
        """準備訓練資料，包括特徵工程和特徵選擇"""
        print("準備訓練資料...")
        if self.train_data is None:
            print("錯誤：訓練資料未載入。請先呼叫 load_data()。")
            return

        self.processed_train_df = self.preprocess_features(self.train_data, is_train=True)

        # 選擇用於模型訓練的特徵欄位
        # 排除原始ID、時間戳、目標變數、以及可能引起洩漏或非數值的欄位
        exclude_cols = ['bid_id', 'timestamp', 'paying_price', 'click', 'ip', 'user_tags', 'key_page_url']
        # 如果 'key_page_url_processed' 存在，則使用它，原始的 'key_page_url' 已被排除
        
        potential_feature_cols = [col for col in self.processed_train_df.columns if col not in exclude_cols]
        
        # 確保所有選定的特徵都是數值型
        self.feature_columns = []
        for col in potential_feature_cols:
            if pd.api.types.is_numeric_dtype(self.processed_train_df[col]):
                self.feature_columns.append(col)
            else:
                print(f"警告：特徵 '{col}' (類型: {self.processed_train_df[col].dtype}) 不是數值型，將被排除。")

        if not self.feature_columns:
            print("錯誤：沒有可用的數值型特徵進行訓練。")
            return

        self.X_train = self.processed_train_df[self.feature_columns]
        self.y_ctr = self.processed_train_df['click'].astype(int) # 確保 click 是整數型

        print(f"選定的特徵數量: {len(self.feature_columns)}")
        print(f"訓練樣本數: {len(self.X_train)}")
        print("click 標籤分布：")
        print(self.y_ctr.value_counts(normalize=True))


    def train_ctr_model(self):
        """訓練 CTR 預測模型 (整合優化方法)"""
        print("訓練 CTR 模型...")
        if self.X_train is None or self.y_ctr is None:
            print("錯誤：訓練資料 (X_train 或 y_ctr) 未準備好。請先呼叫 prepare_training_data()。")
            return

        # 1. 印出完整的類別不平衡情況
        neg_count = sum(self.y_ctr == 0)
        pos_count = sum(self.y_ctr == 1)
        print(f"類別不平衡分析：")
        print(f"- 負樣本 (未點擊): {neg_count} ({neg_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 正樣本 (已點擊): {pos_count} ({pos_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 不平衡比率: {neg_count/pos_count:.1f}:1")

        # 2. 設定更好的參數處理不平衡資料
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc', 'average_precision'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'scale_pos_weight': neg_count/pos_count,  # 只保留這個
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        # 3. 切分資料 (使用分層抽樣確保正樣本比例一致)
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_ctr, test_size=0.2, random_state=42, stratify=self.y_ctr
        )

        # 4. 決定是否需要資料採樣
        use_sampling = False  # 設為 True 啟用資料採樣
        if use_sampling and pos_count / len(self.y_ctr) < 0.01:  # 如果正樣本比例過低才採樣
            print("執行資料採樣，平衡正負樣本比例...")
            # 方法一：欠採樣 (簡單隨機抽樣)
            pos_indices = np.where(y_tr == 1)[0]
            neg_indices = np.where(y_tr == 0)[0]
            
            # 採樣負樣本，保持 10:1 的比例
            target_ratio = 10  # 負:正 = 10:1
            sampled_neg_indices = np.random.choice(
                neg_indices, 
                size=min(len(neg_indices), len(pos_indices) * target_ratio), 
                replace=False
            )
            
            # 合併正樣本和採樣後的負樣本
            sampled_indices = np.concatenate([pos_indices, sampled_neg_indices])
            X_tr_sampled = X_tr.iloc[sampled_indices]
            y_tr_sampled = y_tr.iloc[sampled_indices]
            
            # 使用採樣後的資料
            train_data = lgb.Dataset(X_tr_sampled, label=y_tr_sampled)
            print(f"採樣後訓練集大小: {len(X_tr_sampled)}, 正樣本比例: {sum(y_tr_sampled)/len(y_tr_sampled)*100:.2f}%")
        else:
            # 使用原始資料
            train_data = lgb.Dataset(X_tr, label=y_tr)

        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 5. 訓練模型
        self.ctr_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )

        # 6. 特徵重要性分析
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.ctr_model.feature_importance()
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\n前10個最重要特徵:")
        print(feature_importance.head(10))
        
        # 7. 儲存最重要的特徵 (可用於縮減特徵空間)
        important_threshold = 10  # 只保留重要性大於閾值的特徵
        self.important_features = feature_importance[
            feature_importance['importance'] > important_threshold
        ]['feature'].tolist()
        
        print(f"\n重要特徵 (重要性 > {important_threshold}) 數量: {len(self.important_features)}")
        
        # 8. 計算驗證集上的評估指標
        y_pred_val = self.ctr_model.predict(X_val)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y_val, y_pred_val)
        ap = average_precision_score(y_val, y_pred_val)
        
        print(f"\n驗證集評估指標:")
        print(f"- AUC: {auc:.4f}")
        print(f"- Average Precision: {ap:.4f}")
        
        print("\nCTR 模型訓練完成。")
        
        # 選擇性: 僅使用重要特徵重訓練 (如果特徵數量大幅減少)
        if len(self.important_features) > 5 and len(self.important_features) < len(self.feature_columns) / 2:
            use_important_features_only = False  # 設為 True 啟用重要特徵重訓練
            if use_important_features_only:
                print("\n使用重要特徵重訓練模型...")
                X_tr_important = X_tr[self.important_features]
                X_val_important = X_val[self.important_features]
                
                train_data_important = lgb.Dataset(X_tr_important, label=y_tr)
                valid_data_important = lgb.Dataset(X_val_important, label=y_val, reference=train_data_important)
                
                self.ctr_model = lgb.train(
                    params,
                    train_data_important,
                    valid_sets=[train_data_important, valid_data_important],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
                )
                
                # 更新 feature_columns 為重要特徵
                self.ctr_feature_columns = self.important_features.copy()
                print("使用重要特徵重訓練完成。")


    def train_winprice_model(self):
        """訓練 Win-Price 預測模型 (使用生存分析)"""
        print("訓練 Win-Price 模型...")
        if self.train_data is None or self.processed_train_df is None or self.feature_columns is None:
            print("錯誤：訓練 Win-Price 模型所需的資料未準備好。")
            return

        # 1. 準備生存分析所需的 duration 和 event 欄位 (向量化操作)
        survival_base = pd.DataFrame({'bid_id': self.train_data['bid_id']})
        survival_base['event'] = np.where(
            (pd.notna(self.train_data['paying_price'])) & (self.train_data['paying_price'] > 0), 1, 0
        )
        survival_base['duration'] = np.where(
            survival_base['event'] == 1,
            self.train_data['paying_price'],
            self.train_data['bidding_price'].fillna(0) # 若 bidding_price 也可能 NaN，則填充
        )
        
        # 處理 duration 可能為負值或極小值的情況
        survival_base.loc[survival_base['duration'] <= 0, 'duration'] = 1e-6 # 設為一個極小的正數

        # 2. 合併特徵 (從已處理的 self.processed_train_df)
        # 確保 'bid_id' 在 self.processed_train_df 中以便合併
        if 'bid_id' not in self.processed_train_df.columns:
             # 如果原始 train_data 的索引就是 bid_id，或者可以從原始 train_data 獲取
             # 這裡假設 self.train_data 包含 bid_id，並且與 self.processed_train_df 的行對應
             # 為了安全，最好在 preprocess_features 中保留 bid_id，或確保可以合併
            print("警告：processed_train_df 中缺少 'bid_id'，嘗試從原始 train_data 添加。")
            # 這裡假設 self.processed_train_df 的索引與 self.train_data['bid_id'] 對應
            # 這是一個簡化處理，實際情況可能需要更精確的對齊
            if len(self.processed_train_df) == len(self.train_data):
                 self.processed_train_df['bid_id'] = self.train_data['bid_id'].values
            else:
                 print("錯誤：無法安全地將 'bid_id' 添加到 processed_train_df。")
                 return

        features_to_merge = self.processed_train_df[['bid_id'] + self.feature_columns]
        survival_df = pd.merge(survival_base, features_to_merge, on='bid_id', how='left')

        # 3. 計算 log_duration
        survival_df['log_duration'] = np.log(survival_df['duration']) # duration 已處理為正數

        # 4. 選擇用於 Win-Price 模型的特徵
        # 可以選擇與 CTR 模型不同的特徵子集，或相同的
        self.winprice_features = self.feature_columns[:15] # 例如選擇前15個特徵
        # 確保 'bidding_price' (如果它在 feature_columns 中) 不被用於預測它自己
        if 'bidding_price' in self.winprice_features:
            self.winprice_features.remove('bidding_price')
        
        if not self.winprice_features:
            print("錯誤：沒有為 Win-Price 模型選擇任何特徵。")
            return

        # 5. 檢查並處理 NaN 值 (在傳遞給 lifelines 之前)
        cols_for_lifelines = ['log_duration', 'event'] + self.winprice_features
        final_survival_df_for_fit = survival_df[cols_for_lifelines].copy()

        for col in cols_for_lifelines:
            if final_survival_df_for_fit[col].isnull().any():
                print(f"警告：Win-Price 模型的欄位 '{col}' 中存在 NaN 值。")
                if pd.api.types.is_numeric_dtype(final_survival_df_for_fit[col]):
                    fill_value = final_survival_df_for_fit[col].median() # 或 mean()
                    if pd.isna(fill_value): fill_value = 0 # 如果中位數也是 NaN
                    print(f"正在用中位數/0 ({fill_value}) 填充 '{col}' 中的 NaN。")
                    final_survival_df_for_fit[col].fillna(fill_value, inplace=True)
                else: # 理論上都應該是數值型了
                    print(f"欄位 '{col}' 不是數值型且有 NaN，無法自動填充。")
                    # 可能需要移除這些行或更特定的處理
                    final_survival_df_for_fit.dropna(subset=[col], inplace=True)
        
        # 移除仍然包含 NaN/inf 的行 (最後的保險)
        final_survival_df_for_fit.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_survival_df_for_fit.dropna(inplace=True)

        if final_survival_df_for_fit.empty:
            print("錯誤：處理 NaN 後，Win-Price 模型沒有可用的訓練資料。")
            return
        
        # 6. 訓練 Weibull AFT 模型
        try:
            aft = WeibullAFTFitter()
            aft.fit(final_survival_df_for_fit, duration_col='log_duration', event_col='event')
            self.winprice_model = aft
            print("Win-Price 模型訓練完成。")
        except Exception as e:
            print(f"訓練 Win-Price 模型時發生嚴重錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.winprice_model = None


    def predict_CTR(self, row_series):
        """預測單筆資料的 CTR"""
        if self.ctr_model is None:
            # print("警告：CTR 模型未訓練，返回預設 pctr_min。")
            return self.pctr_min
        
        try:
            # 將 Series 轉換為 DataFrame 以便 preprocess_features 處理
            df_row = pd.DataFrame([row_series])
            processed_df_row = self.preprocess_features(df_row, is_train=False)
            
            # 確保特徵順序和數量與訓練時一致
            features_for_prediction = processed_df_row[self.feature_columns]
            
            pred = self.ctr_model.predict(features_for_prediction, num_iteration=self.ctr_model.best_iteration)[0]
            return max(float(pred), self.pctr_min) # 確保是 float
        except Exception as e:
            # print(f"CTR 預測錯誤: {e}。返回預設 pctr_min。")
            return self.pctr_min

    def predict_winprice(self, row_series):
        """預測單筆資料的勝價"""
        if self.winprice_model is None or not self.winprice_features:
            # print("警告：Win-Price 模型未訓練或特徵未設定，返回基於 bidding_price 的估計。")
            # 使用一個簡單的備用策略，例如出價的某個百分比
            return max(int(row_series.get('bidding_price', 10) * 0.7), 1) # 假設 bidding_price 存在
        
        try:
            df_row = pd.DataFrame([row_series])
            processed_df_row = self.preprocess_features(df_row, is_train=False)
            
            # 確保特徵與 Win-Price 模型訓練時一致
            features_for_prediction = processed_df_row[self.winprice_features]

            # lifelines 期望 DataFrame
            log_pred_duration = self.winprice_model.predict_expectation(features_for_prediction)[0]
            predicted_price = np.exp(log_pred_duration)
            
            return max(int(predicted_price), 1) # 返回整數價格，至少為1
        except Exception as e:
            # print(f"Win-Price 預測錯誤: {e}。返回基於 bidding_price 的估計。")
            return max(int(row_series.get('bidding_price', 10) * 0.7), 1)


    def predict_CTR_from_processed(self, processed_features_row):
        """從已處理的單行特徵預測 CTR"""
        if self.ctr_model is None:
            return self.pctr_min
        try:
            # processed_features_row 應該是一個包含 self.feature_columns 的 Pandas Series 或 DataFrame 行
            # 確保傳入的特徵與訓練時的 feature_columns 順序和名稱一致
            pred = self.ctr_model.predict(pd.DataFrame([processed_features_row[self.feature_columns]]), num_iteration=self.ctr_model.best_iteration)[0]
            return max(float(pred), self.pctr_min)
        except Exception as e:
            # print(f"CTR 預測 (processed) 錯誤: {e}")
            return self.pctr_min

    def predict_winprice_from_processed(self, processed_features_row):
        """從已處理的單行特徵預測勝價"""
        if self.winprice_model is None or not self.winprice_features:
            return max(int(processed_features_row.get('bidding_price', 10) * 0.7), 1) # bidding_price 可能不在 processed_features_row
        try:
            # processed_features_row 應該是一個包含 self.winprice_features 的 Pandas Series 或 DataFrame 行
            log_pred_duration = self.winprice_model.predict_expectation(pd.DataFrame([processed_features_row[self.winprice_features]]))[0]
            predicted_price = np.exp(log_pred_duration)
            return max(int(predicted_price), 1)
        except Exception as e:
            # print(f"Win-Price 預測 (processed) 錯誤: {e}")
            # 需要一個備用策略，如果 processed_features_row 沒有 bidding_price
            # 可以這樣傳遞: self.predict_winprice_from_processed(processed_row, original_row.get('bidding_price'))
            # 但為了簡化，假設 predict_winprice_from_processed 內部有備用邏輯
            # 返回一個較小值或基於其他可用資訊
            return 1 # 或者更智能的備用值


    def bid_day1(self):
        """執行 Day1 的出價邏輯"""
        print("執行 Day1 出價...")
        if self.test_day1 is None:
            print("錯誤：測試資料未載入。")
            return None
        if self.ctr_model is None or self.winprice_model is None:
            print("錯誤：一個或多個模型未訓練。無法執行出價。")
            return None

        remaining_budget = self.DAY_BUDGET
        spent_per_hour = [0] * 24
        bid_results = []
        
        # --- 優化點：預處理整個測試集 ---
        print("預處理整個測試集進行出價...")
        processed_test_df = self.preprocess_features(self.test_day1.copy(), is_train=False) # 使用副本
        print("測試集預處理完成。")

        # 確保 CTR 模型和 Win-Price 模型所需的特徵都存在於 processed_test_df
        missing_ctr_cols = [col for col in self.feature_columns if col not in processed_test_df.columns]
        if missing_ctr_cols:
            print(f"錯誤：預處理後的測試集缺少 CTR 模型所需的特徵: {missing_ctr_cols}")
            return None
        
        missing_wp_cols = [col for col in self.winprice_features if col not in processed_test_df.columns]
        if missing_wp_cols:
            print(f"錯誤：預處理後的測試集缺少 Win-Price 模型所需的特徵: {missing_wp_cols}")
            return None
        # --- 優化點結束 ---

        num_bids_made = 0
        total_spent_if_won = 0

        # 使用 processed_test_df 進行迭代
        # 為了能同時訪問原始 test_day1 的 'timestamp' (如果 preprocess_features 移除了它)
        # 和 processed_test_df 的特徵，可以考慮合併或使用索引對齊
        # 這裡假設 'hour' 已經在 processed_test_df 中被正確產生和保留
        # 並且 'bid_id' 也被保留或可以從索引獲得

        for idx in range(len(processed_test_df)):
            processed_row = processed_test_df.iloc[idx] # 獲取已處理的行 (Pandas Series)
            original_row = self.test_day1.iloc[idx] # 獲取原始行，以備不時之需 (例如原始 bidding_price)

            current_hour = int(processed_row.get('hour', 0)) # 假設 'hour' 在 processed_row 中
            bid_id = original_row.get('bid_id', f"unknown_bid_{idx}") # 從原始資料獲取 bid_id
            
            bid_price_for_this_impression = 0

            if spent_per_hour[current_hour] >= self.hourly_budget[current_hour] or remaining_budget <= 0:
                bid_price_for_this_impression = 0
            else:
                # --- 修改：使用新的預測方法 ---
                predicted_ctr = self.predict_CTR_from_processed(processed_row)

                if predicted_ctr < self.pctr_min:
                    bid_price_for_this_impression = 0
                else:
                    # 如果 predict_winprice_from_processed 需要原始 bidding_price 作為備用
                    # 可以這樣傳遞: self.predict_winprice_from_processed(processed_row, original_row.get('bidding_price'))
                    # 但為了簡化，假設 predict_winprice_from_processed 內部有備用邏輯
                    predicted_win_price = self.predict_winprice_from_processed(processed_row)
                    # --- 修改結束 ---

                    if predicted_win_price <= 0:
                        bid_price_for_this_impression = 0
                    else:
                        potential_bid = predicted_win_price + 1 

                        if remaining_budget >= potential_bid and \
                           (spent_per_hour[current_hour] + potential_bid) <= self.hourly_budget[current_hour]:
                            bid_price_for_this_impression = potential_bid
                        else:
                            bid_price_for_this_impression = 0
            
            if bid_price_for_this_impression > 0:
                remaining_budget -= bid_price_for_this_impression
                spent_per_hour[current_hour] += bid_price_for_this_impression
                num_bids_made += 1
                total_spent_if_won += bid_price_for_this_impression

            bid_results.append({
                'bid_id': bid_id,
                'paying_price': bid_price_for_this_impression
            })

            if (idx + 1) % 50000 == 0:
                print(f"已處理 {idx + 1}/{len(processed_test_df)} 筆競價請求. "
                      f"剩餘總預算: {remaining_budget:.2f}. "
                      f"出價次數: {num_bids_made}. "
                      f"假設花費: {total_spent_if_won:.2f}")

        result_df = pd.DataFrame(bid_results)
        output_filename = f"{self.student_id}_day1.csv"
        try:
            result_df.to_csv(output_filename, index=False)
            print(f"\nDay1 出價完成，結果已儲存至: {output_filename}")
            print(f"總出價次數 (paying_price > 0): {(result_df['paying_price'] > 0).sum()}")
            print(f"總出價金額 (假設都贏得且按此價格支付): {result_df['paying_price'].sum()}")
            print(f"剩餘總預算: {remaining_budget:.2f}")
        except Exception as e:
            print(f"儲存出價結果時發生錯誤: {e}")
        
        return result_df

# --- 以下是執行流程控制 (類似 run.py 的功能) ---

def check_data_files():
    """檢查必要的資料檔案是否存在"""
    required_files = [
        'data/train.csv',
        'data/test_day1.csv'
    ]
    # 可選: 'data/Feature_Meaning.xlsx'
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("錯誤：缺少以下必要的資料檔案:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    return True

def run_rtb_pipeline():
    """執行完整的 RTB 競價系統流程"""
    print("=== RTB 競價系統執行開始 ===")
    
    if not check_data_files():
        print("由於資料檔案缺失，執行中止。")
        return

    student_id = "M36134016" # 請替換為您的學號

    try:
        # 1. 初始化系統
        print("\n--- 步驟 1: 初始化系統 ---")
        rtb_system = RTBBiddingSystem(student_id=student_id)
        
        # 2. 載入資料
        print("\n--- 步驟 2: 載入資料 ---")
        rtb_system.load_data()
        
        # 3. 特徵工程與訓練資料準備
        print("\n--- 步驟 3: 特徵工程與訓練資料準備 ---")
        rtb_system.prepare_training_data()
        
        # 4. 訓練 CTR 模型
        print("\n--- 步驟 4: 訓練 CTR 模型 ---")
        rtb_system.train_ctr_model()
        
        # 5. 訓練 Win-Price 模型
        print("\n--- 步驟 5: 訓練 Win-Price 模型 ---")
        rtb_system.train_winprice_model()
        
        # 6. Day1 出價
        print("\n--- 步驟 6: 執行 Day1 出價 ---")
        rtb_system.bid_day1()
        
        print(f"\n=== RTB 競價系統執行完成 ===")
        
    except FileNotFoundError:
        # load_data 內部已處理，這裡再次捕獲以防萬一
        print("執行因檔案未找到而中止。")
    except Exception as e:
        print(f"\n執行過程中發生未預期的錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_rtb_pipeline()