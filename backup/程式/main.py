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

# DeepCTR 相關導入
try:
    from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
    from deepctr.models import DeepFM, xDeepFM
    import tensorflow as tf
    DEEPCTR_AVAILABLE = True
    print("✅ DeepCTR 可用")
except ImportError as e:
    print(f"⚠️ DeepCTR 未安裝或導入失敗: {e}")
    print("請執行: pip install deepctr tensorflow")
    DEEPCTR_AVAILABLE = False

warnings.filterwarnings('ignore')

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
    def train_ctr_models(self):
        """訓練三個 CTR 預測模型 (LightGBM + DeepFM + xDeepFM) 並比較視覺化"""
        print("🚀 開始訓練三個 CTR 模型...")
        if self.X_train is None or self.y_ctr is None:
            print("錯誤：訓練資料 (X_train 或 y_ctr) 未準備好。請先呼叫 prepare_training_data()。")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        
        # 使用統一的時間戳建立資料夾
        viz_dir = f"ctr_models_comparison_{self.timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 類別不平衡分析
        neg_count = sum(self.y_ctr == 0)
        pos_count = sum(self.y_ctr == 1)
        print(f"類別不平衡分析：")
        print(f"- 負樣本 (未點擊): {neg_count} ({neg_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 正樣本 (已點擊): {pos_count} ({pos_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 不平衡比率: {neg_count/pos_count:.1f}:1")
        
        # 2. 切分資料
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_ctr, test_size=0.2, random_state=42, stratify=self.y_ctr
        )
        
        # 儲存模型評估結果
        model_results = {}
        
        # ========== 1. 訓練 LightGBM 模型 ==========
        print("\n🔵 訓練 LightGBM 模型...")
        
        lgb_params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'verbose': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'min_data_in_leaf': 50,
            'scale_pos_weight': neg_count/pos_count,
            'n_jobs': -1,
            'seed': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.ctr_model_lgb = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(0)]
        )
        
        y_pred_lgb = self.ctr_model_lgb.predict(X_val)
        model_results['LightGBM'] = {
            'predictions': y_pred_lgb,
            'auc': roc_auc_score(y_val, y_pred_lgb),
            'ap': average_precision_score(y_val, y_pred_lgb)
        }
        print(f"✅ LightGBM - AUC: {model_results['LightGBM']['auc']:.4f}, AP: {model_results['LightGBM']['ap']:.4f}")
        
        # ========== 2. 訓練 DeepFM 和 xDeepFM (如果可用) ==========
        if DEEPCTR_AVAILABLE:
            print("\n🟢 準備 DeepCTR 資料...")
            
            # 準備 DeepCTR 格式資料
            deepctr_data = self.processed_train_df.copy()
            
            # 分離類別特徵和數值特徵
            categorical_features = []
            numerical_features = []
            
            for col in self.feature_columns:
                if deepctr_data[col].dtype == 'object' or deepctr_data[col].nunique() < 100:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            
            # 類別特徵編碼
            for feat in categorical_features:
                if feat not in self.deepctr_scalers:
                    lbe = LabelEncoder()
                    deepctr_data[feat] = lbe.fit_transform(deepctr_data[feat].astype(str))
                    self.deepctr_scalers[f'{feat}_encoder'] = lbe
            
            # 數值特徵標準化
            if numerical_features:
                if 'numerical_scaler' not in self.deepctr_scalers:
                    mms = MinMaxScaler()
                    deepctr_data[numerical_features] = mms.fit_transform(deepctr_data[numerical_features])
                    self.deepctr_scalers['numerical_scaler'] = mms
            
            # 建立特徵欄位描述
            fixlen_feature_columns = []
            
            for feat in categorical_features:
                vocab_size = deepctr_data[feat].nunique()
                fixlen_feature_columns.append(
                    SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=min(4, vocab_size))
                )
            
            for feat in numerical_features:
                fixlen_feature_columns.append(DenseFeat(feat, 1))
            
            self.deepctr_feature_columns = fixlen_feature_columns
            self.deepctr_feature_names = get_feature_names(fixlen_feature_columns)
            
            # 切分資料
            deepctr_train, deepctr_valid = train_test_split(
                deepctr_data, test_size=0.2, random_state=42, stratify=deepctr_data['click']
            )
            
            train_model_input = {name: deepctr_train[name].values for name in self.deepctr_feature_names}
            valid_model_input = {name: deepctr_valid[name].values for name in self.deepctr_feature_names}
            
            # 訓練 DeepFM
            print("🟢 訓練 DeepFM 模型...")
            try:
                self.ctr_model_deepfm = DeepFM(fixlen_feature_columns, fixlen_feature_columns, task='binary')
                self.ctr_model_deepfm.compile("adam", "binary_crossentropy", metrics=['AUC'])
                
                history_deepfm = self.ctr_model_deepfm.fit(
                    train_model_input, deepctr_train['click'].values,
                    batch_size=1024, epochs=10, verbose=0, validation_split=0.1
                )
                
                y_pred_deepfm = self.ctr_model_deepfm.predict(valid_model_input, batch_size=256).flatten()
                model_results['DeepFM'] = {
                    'predictions': y_pred_deepfm,
                    'auc': roc_auc_score(deepctr_valid['click'].values, y_pred_deepfm),
                    'ap': average_precision_score(deepctr_valid['click'].values, y_pred_deepfm)
                }
                print(f"✅ DeepFM - AUC: {model_results['DeepFM']['auc']:.4f}, AP: {model_results['DeepFM']['ap']:.4f}")
                
            except Exception as e:
                print(f"❌ DeepFM 訓練失敗: {e}")
                model_results['DeepFM'] = None
            
            # 訓練 xDeepFM
            print("🟡 訓練 xDeepFM 模型...")
            try:
                self.ctr_model_xdeepfm = xDeepFM(fixlen_feature_columns, fixlen_feature_columns, task='binary')
                self.ctr_model_xdeepfm.compile("adam", "binary_crossentropy", metrics=['AUC'])
                
                history_xdeepfm = self.ctr_model_xdeepfm.fit(
                    train_model_input, deepctr_train['click'].values,
                    batch_size=1024, epochs=10, verbose=0, validation_split=0.1
                )
                
                y_pred_xdeepfm = self.ctr_model_xdeepfm.predict(valid_model_input, batch_size=256).flatten()
                model_results['xDeepFM'] = {
                    'predictions': y_pred_xdeepfm,
                    'auc': roc_auc_score(deepctr_valid['click'].values, y_pred_xdeepfm),
                    'ap': average_precision_score(deepctr_valid['click'].values, y_pred_xdeepfm)
                }
                print(f"✅ xDeepFM - AUC: {model_results['xDeepFM']['auc']:.4f}, AP: {model_results['xDeepFM']['ap']:.4f}")
                
            except Exception as e:
                print(f"❌ xDeepFM 訓練失敗: {e}")
                model_results['xDeepFM'] = None
                
            # 儲存處理後的資料供預測使用
            self.deepctr_train_processed = deepctr_data
            
        else:
            print("⚠️ DeepCTR 不可用，跳過 DeepFM 和 xDeepFM 訓練")
            model_results['DeepFM'] = None
            model_results['xDeepFM'] = None
        
        # ========== 3. 模型比較視覺化 ==========
        print(f"\n📊 生成模型比較視覺化...")
        
        # 3.1 ROC 曲線比較
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, result) in enumerate(model_results.items()):
            if result is not None:
                if model_name == 'LightGBM':
                    y_true = y_val
                else:
                    y_true = deepctr_valid['click'].values
                
                fpr, tpr, _ = roc_curve(y_true, result['predictions'])
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{model_name} (AUC = {result["auc"]:.4f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('三個模型 ROC 曲線比較', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{viz_dir}/01_roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3.2 PR 曲線比較
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, result) in enumerate(model_results.items()):
            if result is not None:
                if model_name == 'LightGBM':
                    y_true = y_val
                else:
                    y_true = deepctr_valid['click'].values
                
                precision, recall, _ = precision_recall_curve(y_true, result['predictions'])
                plt.plot(recall, precision, color=colors[i], lw=2,
                        label=f'{model_name} (AP = {result["ap"]:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('三個模型 Precision-Recall 曲線比較', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{viz_dir}/02_pr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3.3 預測分布比較
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (model_name, result) in enumerate(model_results.items()):
            if result is not None:
                if model_name == 'LightGBM':
                    y_true = y_val
                else:
                    y_true = deepctr_valid['click'].values
                
                y_pred_pos = result['predictions'][y_true == 1]
                y_pred_neg = result['predictions'][y_true == 0]
                
                axes[i].hist(y_pred_neg, bins=50, alpha=0.7, label='未點擊', color='lightcoral', density=True)
                axes[i].hist(y_pred_pos, bins=50, alpha=0.7, label='已點擊', color='lightblue', density=True)
                axes[i].set_xlabel('預測機率')
                axes[i].set_ylabel('密度')
                axes[i].set_title(f'{model_name} 預測分布')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/03_prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3.4 模型性能總結表
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        summary_data = []
        for model_name, result in model_results.items():
            if result is not None:
                summary_data.append([model_name, f"{result['auc']:.4f}", f"{result['ap']:.4f}"])
        
        table = plt.table(cellText=summary_data,
                         colLabels=['模型', 'AUC', 'Average Precision'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('模型性能比較總結', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{viz_dir}/04_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 儲存訓練報告
        report = f"""
        CTR 模型比較訓練報告
        ==================

        資料統計：
        - 總樣本數: {len(self.y_ctr):,}
        - 正樣本數: {pos_count:,} ({pos_count/len(self.y_ctr)*100:.2f}%)
        - 負樣本數: {neg_count:,} ({neg_count/len(self.y_ctr)*100:.2f}%)
        - 不平衡比率: {neg_count/pos_count:.1f}:1

        模型性能比較：
        """
        
        for model_name, result in model_results.items():
            if result is not None:
                report += f"""
        {model_name}:
        - AUC: {result['auc']:.4f}
        - Average Precision: {result['ap']:.4f}
        """
        
        report += f"""
        
        最佳模型: {max([k for k, v in model_results.items() if v is not None], 
                      key=lambda x: model_results[x]['auc'])}
        
        視覺化檔案已儲存至: {viz_dir}/
        """
        
        with open(f'{viz_dir}/training_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 三個 CTR 模型訓練完成!")
        print(f"📊 視覺化檔案已儲存至: {viz_dir}/")
        print(f"📄 訓練報告已儲存: {viz_dir}/training_report.txt")
        
        # 選擇最佳模型作為預設
        best_model = max([k for k, v in model_results.items() if v is not None], 
                        key=lambda x: model_results[x]['auc'])
        self.active_ctr_model = best_model.lower().replace('lightgbm', 'lgb').replace('deepfm', 'deepfm').replace('xdeepfm', 'xdeepfm')
        print(f"🎯 設定最佳模型為預設: {best_model}")
    
    def predict_ctr(self, row_series, model_type=None):
        """預測單筆資料的 CTR，支援選擇模型類型"""
        if model_type is None:
            model_type = self.active_ctr_model
            
        try:
            if model_type == 'lgb' and self.ctr_model_lgb is not None:
                # LightGBM 預測
                df_row = pd.DataFrame([row_series])
                processed_df_row = self.preprocess_features(df_row, is_train=False)
                features_for_prediction = processed_df_row[self.feature_columns]
                pred = self.ctr_model_lgb.predict(features_for_prediction, num_iteration=self.ctr_model_lgb.best_iteration)[0]
                return max(float(pred), self.pctr_min)
                
            elif model_type in ['deepfm', 'xdeepfm'] and DEEPCTR_AVAILABLE:
                # DeepCTR 模型預測
                df_row = pd.DataFrame([row_series])
                processed_df_row = self.preprocess_features(df_row, is_train=False)
                
                # 應用相同的預處理
                for feat in self.deepctr_feature_names:
                    if feat in processed_df_row.columns:
                        if f'{feat}_encoder' in self.deepctr_scalers:
                            lbe = self.deepctr_scalers[f'{feat}_encoder']
                            processed_df_row[feat] = lbe.transform(processed_df_row[feat].astype(str))
                
                # 數值特徵標準化
                if 'numerical_scaler' in self.deepctr_scalers:
                    numerical_features = [col for col in self.deepctr_feature_names 
                                        if col in processed_df_row.columns and 
                                        processed_df_row[col].dtype in ['float64', 'int64']]
                    if numerical_features:
                        mms = self.deepctr_scalers['numerical_scaler']
                        processed_df_row[numerical_features] = mms.transform(processed_df_row[numerical_features])
                
                model_input = {name: processed_df_row[name].values for name in self.deepctr_feature_names 
                              if name in processed_df_row.columns}
                
                if model_type == 'deepfm' and self.ctr_model_deepfm is not None:
                    pred = self.ctr_model_deepfm.predict(model_input)[0]
                elif model_type == 'xdeepfm' and self.ctr_model_xdeepfm is not None:
                    pred = self.ctr_model_xdeepfm.predict(model_input)[0]
                else:
                    return self.pctr_min
                    
                return max(float(pred), self.pctr_min)
            else:
                return self.pctr_min
                
        except Exception as e:
            print(f"CTR 預測錯誤 ({model_type}): {e}")
            return self.pctr_min
        
    def train_winprice_model(self):
        """訓練 Win-Price 預測模型 (使用生存分析) - 完整修復版本"""
        print("訓練 Win-Price 模型...")
        if self.train_data is None:
            print("錯誤：訓練資料未載入。請先呼叫 load_data()。")
            return
        
        # 使用 Weibull 分佈擬合 win_price
        if 'win_price' not in self.train_data.columns:
            print("錯誤：訓練資料中缺少 'win_price' 欄位。")
            return
        
        # 只使用有效的數值 (排除負值和 NaN)
        valid_prices = self.train_data[self.train_data['win_price'] > 0]['win_price']
        if valid_prices.empty:
            print("錯誤：沒有有效的 win_price 數據進行擬合。")
            return
        
        # 擬合 Weibull 分佈
        shape, loc, scale = WeibullAFTFitter().fit_right_censoring(
            valid_prices, 
            duration_col=None, 
            event_col=None, 
            show_warnings=False
        )
        
        self.winprice_model = WeibullAFTFitter()
        self.winprice_model.params_ = {
            'lambda': shape,
            'log(scale)': np.log(scale),
            'log(loc)': np.log(loc) if loc > 0 else -np.inf
        }
        self.winprice_model.fitted_ = True
        
        print("✅ Win-Price 模型訓練完成。")
        print(f"- 形狀參數 (lambda): {shape:.4f}")
        print(f"- 尺度參數 (scale): {scale:.4f}")
        print(f"- 位移參數 (loc): {loc:.4f}")

    def predict_winprice(self, row_series):
        """預測單筆資料的勝價 - 極度保守版本"""
        if self.winprice_model is None:
            print("錯誤：Win-Price 模型未訓練。")
            return 0
        
        # 使用 Weibull 分佈的逆累積分佈函數 (Percent-Point Function, PPF)
        shape = self.winprice_model.params_['lambda']
        scale = np.exp(self.winprice_model.params_['log(scale)'])
        loc = np.exp(self.winprice_model.params_['log(loc)']) if 'log(loc)' in self.winprice_model.params_ else 0
        
        # 特徵處理
        df_row = pd.DataFrame([row_series])
        processed_df_row = self.preprocess_features(df_row, is_train=False)
        
        # 使用所有 winprice_features 作為預測依據
        features_for_prediction = processed_df_row[self.winprice_features].fillna(0)
        
        # 簡化預測邏輯：直接使用特徵的線性組合
        linear_combination = np.dot(features_for_prediction, self.winprice_model.params_[1:]) + self.winprice_model.params_[0]
        
        # 使用 PPF 進行預測
        pred = np.maximum(0, linear_combination)  # 勝價不能為負
        return float(pred)

    def predict_CTR_from_processed(self, processed_features_row):
        """從已處理的單行特徵預測 CTR"""
        if self.active_ctr_model == 'lgb' and self.ctr_model_lgb is not None:
            features_for_prediction = processed_features_row[self.feature_columns]
            pred = self.ctr_model_lgb.predict(features_for_prediction, num_iteration=self.ctr_model_lgb.best_iteration)
            return max(float(pred), self.pctr_min)
        elif self.active_ctr_model in ['deepfm', 'xdeepfm'] and DEEPCTR_AVAILABLE:
            df_row = pd.DataFrame([processed_features_row])
            processed_df_row = self.preprocess_features(df_row, is_train=False)
            
            # 應用相同的預處理
            for feat in self.deepctr_feature_names:
                if feat in processed_df_row.columns:
                    if f'{feat}_encoder' in self.deepctr_scalers:
                        lbe = self.deepctr_scalers[f'{feat}_encoder']
                        processed_df_row[feat] = lbe.transform(processed_df_row[feat].astype(str))
            
            # 數值特徵標準化
            if 'numerical_scaler' in self.deepctr_scalers:
                numerical_features = [col for col in self.deepctr_feature_names 
                                    if col in processed_df_row.columns and 
                                    processed_df_row[col].dtype in ['float64', 'int64']]
                if numerical_features:
                    mms = self.deepctr_scalers['numerical_scaler']
                    processed_df_row[numerical_features] = mms.transform(processed_df_row[numerical_features])
            
            model_input = {name: processed_df_row[name].values for name in self.deepctr_feature_names 
                          if name in processed_df_row.columns}
            
            if self.active_ctr_model == 'deepfm' and self.ctr_model_deepfm is not None:
                pred = self.ctr_model_deepfm.predict(model_input)
            elif self.active_ctr_model == 'xdeepfm' and self.ctr_model_xdeepfm is not None:
                pred = self.ctr_model_xdeepfm.predict(model_input)
            else:
                return self.pctr_min
                
            return max(float(pred), self.pctr_min)
        else:
            return self.pctr_min

    def predict_winprice_from_processed(self, processed_features_row):
        """從已處理的單行特徵預測勝價"""
        if self.winprice_model is None:
            print("錯誤：Win-Price 模型未訓練。")
            return 0
        
        # 使用 Weibull 分佈的逆累積分佈函數 (Percent-Point Function, PPF)
        shape = self.winprice_model.params_['lambda']
        scale = np.exp(self.winprice_model.params_['log(scale)'])
        loc = np.exp(self.winprice_model.params_['log(loc)']) if 'log(loc)' in self.winprice_model.params_ else 0
        
        # 特徵處理
        df_row = pd.DataFrame([processed_features_row])
        processed_df_row = self.preprocess_features(df_row, is_train=False)
        
        # 使用所有 winprice_features 作為預測依據
        features_for_prediction = processed_df_row[self.winprice_features].fillna(0)
        
        # 簡化預測邏輯：直接使用特徵的線性組合
        linear_combination = np.dot(features_for_prediction, self.winprice_model.params_[1:]) + self.winprice_model.params_[0]
        
        # 使用 PPF 進行預測
        pred = np.maximum(0, linear_combination)  # 勝價不能為負
        return float(pred)

    def bid_day1(self):
        """執行 Day1 的出價邏輯"""
        print("執行 Day1 出價...")
        if self.test_day2 is None:
            print("錯誤：測試資料未載入。")
            return None
        if self.ctr_model_lgb is None or self.winprice_model is None:
            print("錯誤：一個或多個模型未訓練。無法執行出價。")
            return None

        remaining_budget = self.DAY_BUDGET
        spent_per_hour = [0] * 24
        bid_results = []
        
        # 預處理整個測試集
        print("預處理整個測試集進行出價...")
        processed_test_df = self.preprocess_features(self.test_day2.copy(), is_train=False)
        print("測試集預處理完成。")

        # 批量預測CTR (使用當前活躍模型)
        batch_size = 10000
        if self.active_ctr_model == 'lgb' and self.ctr_model_lgb is not None:
            for i in range(0, len(processed_test_df), batch_size):
                batch = processed_test_df.iloc[i:i+batch_size]
                ctr_preds = self.ctr_model_lgb.predict(
                    batch[self.feature_columns], 
                    num_iteration=self.ctr_model_lgb.best_iteration
                )
                processed_test_df.loc[batch.index, 'predicted_ctr'] = ctr_preds
        elif self.active_ctr_model in ['deepfm', 'xdeepfm'] and DEEPCTR_AVAILABLE:
            # DeepCTR 批量預測邏輯
            model = self.ctr_model_deepfm if self.active_ctr_model == 'deepfm' else self.ctr_model_xdeepfm
            if model is not None:
                for i in range(0, len(processed_test_df), batch_size):
                    batch = processed_test_df.iloc[i:i+batch_size]
                    model_input = {name: batch[name].values for name in self.deepctr_feature_names 
                                  if name in batch.columns}
                    ctr_preds = model.predict(model_input, batch_size=min(batch_size, len(batch))).flatten()
                    processed_test_df.loc[batch.index, 'predicted_ctr'] = ctr_preds
        
        # 檢查CTR預測結果
        print(f"CTR預測結果統計 ({self.active_ctr_model}):")
        print(f"- 平均值: {processed_test_df['predicted_ctr'].mean()}")
        print(f"- 最小值: {processed_test_df['predicted_ctr'].min()}")
        print(f"- 最大值: {processed_test_df['predicted_ctr'].max()}")
        print(f"- 高於閾值({self.pctr_min})的比例: {(processed_test_df['predicted_ctr'] > self.pctr_min).mean()*100:.2f}%")
        
        # 批量預測win_price
        batch_size = 10000
        for i in range(0, len(processed_test_df), batch_size):
            batch = processed_test_df.iloc[i:i+batch_size]
            batch_winprice = batch[self.winprice_features].copy()
            batch_winprice = batch_winprice.fillna(0)
            win_price_preds = self.winprice_model.predict_expectation(batch_winprice)
            processed_test_df.loc[batch.index, 'predicted_win_price'] = np.exp(win_price_preds)
        
        # 檢查win_price預測結果
        print(f"Win-Price預測結果統計:")
        print(f"- 平均值: {processed_test_df['predicted_win_price'].mean()}")
        print(f"- 最小值: {processed_test_df['predicted_win_price'].min()}")
        print(f"- 最大值: {processed_test_df['predicted_win_price'].max()}")
        
        # 簡化出價策略，確保有出價
        num_bids_made = 0
        total_spent_if_won = 0

        for idx in range(len(processed_test_df)):
            processed_row = processed_test_df.iloc[idx]
            original_row = self.test_day2.iloc[idx]

            current_hour = int(processed_row.get('hour', 0))
            bid_id = original_row.get('bid_id', f"unknown_bid_{idx}")
            
            bid_price_for_this_impression = 0

            if spent_per_hour[current_hour] >= self.hourly_budget[current_hour] or remaining_budget <= 0:
                bid_price_for_this_impression = 0
            else:
                predicted_ctr = processed_row.get('predicted_ctr', self.pctr_min)
                # 放寬CTR閾值條件
                if predicted_ctr < self.pctr_min * 0.1:  # 降低閾值為原來的10%
                    bid_price_for_this_impression = 0
                else:
                    predicted_win_price = processed_row.get('predicted_win_price', 0)
                    # 修正：處理 NaN 或 inf
                    if not np.isfinite(predicted_win_price) or predicted_win_price <= 0:
                        predicted_win_price = 1  # 給一個最小有效值
                    # 簡化出價策略 - 直接出價 = 預測勝價 + 1
                    potential_bid = max(int(predicted_win_price) + 1, 1)
                    # 檢查預算限制
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

        # ========== 預算補標邏輯：確保預算用完 ==========
        if remaining_budget > 0:
            print(f"\n預算尚餘 {remaining_budget:.2f}，開始補標流程...")
            # 找出未出價的 impression
            unbid_indices = [i for i, r in enumerate(bid_results) if r['paying_price'] == 0]
            # 依 predicted_ctr 由高到低排序
            unbid_sorted = sorted(unbid_indices, key=lambda i: processed_test_df.iloc[i].get('predicted_ctr', 0), reverse=True)
            for i in unbid_sorted:
                if remaining_budget <= 0:
                    break
                processed_row = processed_test_df.iloc[i]
                original_row = self.test_day2.iloc[i]
                current_hour = int(processed_row.get('hour', 0))
                # 只補標小時額度還有空間的
                if spent_per_hour[current_hour] < self.hourly_budget[current_hour]:
                    predicted_win_price = processed_row.get('predicted_win_price', 0)
                    if not np.isfinite(predicted_win_price) or predicted_win_price <= 0:
                        predicted_win_price = 1
                    # 直接 all-in 或平均分配
                    potential_bid = min(remaining_budget, self.hourly_budget[current_hour] - spent_per_hour[current_hour], max(int(predicted_win_price) + 1, 1))
                    if potential_bid > 0:
                        bid_results[i]['paying_price'] = potential_bid
                        remaining_budget -= potential_bid
                        spent_per_hour[current_hour] += potential_bid
            print(f"補標後剩餘預算: {remaining_budget:.2f}")
        # ========== 補標結束 ==========

        # 儲存結果
        result_df = pd.DataFrame(bid_results)
        output_filename = f"{self.student_id}_day2_{self.timestamp}.csv"
        result_df.to_csv(output_filename, index=False)
        
        # 增強統計資訊
        final_bid_count = (result_df['paying_price'] > 0).sum()
        final_total = result_df['paying_price'].sum()
        avg_bid = result_df[result_df['paying_price'] > 0]['paying_price'].mean() if final_bid_count > 0 else 0
        
        print(f"\nDay2 出價完成，結果已儲存至: {output_filename}")
        print(f"使用模型: {self.active_ctr_model}")
        print(f"總出價次數: {final_bid_count}")
        print(f"實際花費: {final_total}")
        print(f"預算利用率: {final_total/self.DAY_BUDGET*100:.1f}%")
        print(f"平均出價: {avg_bid:.2f}")
        
        if final_bid_count > 0:
            bid_range = result_df[result_df['paying_price'] > 0]['paying_price']
            print(f"出價範圍: {bid_range.min()} - {bid_range.max()}")
        
        print(f"剩餘總預算: {remaining_budget:.2f}")
        
        return result_df

# --- 以下是執行流程控制 (類似 run.py 的功能) ---

def check_data_files():
    """檢查必要的資料檔案是否存在"""
    required_files = [
        'data/train.csv',
        'data/test_day2.csv'
    ]
    
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

    student_id = "M36134016"

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
        rtb_system.train_ctr_models()
        
        # 5. 訓練 Win-Price 模型
        print("\n--- 步驟 5: 訓練 Win-Price 模型 ---")
        rtb_system.train_winprice_model()
        
        # 6. Day2 出價
        print("\n--- 步驟 6: 執行 Day2 出價 ---")
        rtb_system.active_ctr_model = 'xdeepfm'  # 或 'lgb', 'deepfm'
        rtb_system.bid_day2()
        
        print(f"\n=== RTB 競價系統執行完成 ===")
        
    except FileNotFoundError:
        print("執行因檔案未找到而中止。")
    except Exception as e:
        print(f"\n執行過程中發生未預期的錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_rtb_pipeline()