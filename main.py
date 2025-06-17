import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lifelines import WeibullAFTFitter
import warnings
import os
from collections import Counter
import datetime
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings('ignore')

class RTBBiddingSystem:
    def __init__(self, student_id="M36134016"):
        self.student_id = student_id
        self.DAY_BUDGET = 5000
        self.pctr_min = 1e-4  # 最低可接受的預測點擊率
        self.rho_cut = 2e-5   # 性價比門檻 (pCTR / win_price)
        
        # 動態調整每小時預算，根據歷史競價情況分配，而不是均分
        hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.3, 1.1, 1.0, 
                        1.0, 1.1, 1.3, 1.5, 1.8, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4]
        total_weight = sum(hour_weights)
        self.hourly_budget = [(self.DAY_BUDGET * w / total_weight) for w in hour_weights]

        # 模型
        self.ctr_model = None
        self.winprice_model = None
        self.feature_encoders = {}  # 儲存 LabelEncoders 和其他轉換器
        self.processed_train_df = None # 儲存處理過的訓練資料以供重用

        # 資料
        self.train_data = None
        self.test_day2 = None
        self.X_train = None
        self.y_ctr = None
        self.feature_columns = None # 用於模型訓練的特徵欄位名稱
        self.winprice_features = None # 用於 Win-Price 模型訓練的特徵

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


    def train_ctr_model(self):
        """訓練 CTR 預測模型 (整合優化方法) - 增強視覺化版本"""
        print("訓練 CTR 模型...")
        if self.X_train is None or self.y_ctr is None:
            print("錯誤：訓練資料 (X_train 或 y_ctr) 未準備好。請先呼叫 prepare_training_data()。")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        import os
        
        # 建立視覺化資料夾
        viz_dir = "ctr_model_visualization"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 類別不平衡分析與視覺化
        neg_count = sum(self.y_ctr == 0)
        pos_count = sum(self.y_ctr == 1)
        print(f"類別不平衡分析：")
        print(f"- 負樣本 (未點擊): {neg_count} ({neg_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 正樣本 (已點擊): {pos_count} ({pos_count/len(self.y_ctr)*100:.2f}%)")
        print(f"- 不平衡比率: {neg_count/pos_count:.1f}:1")
        
        # 📊 視覺化1: 類別分布圓餅圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 圓餅圖
        labels = ['未點擊 (0)', '已點擊 (1)']
        sizes = [neg_count, pos_count]
        colors = ['lightcoral', 'lightblue']
        explode = (0, 0.1)  # 突出正樣本
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
                shadow=True, startangle=90)
        ax1.set_title('CTR 資料類別分布', fontsize=14, fontweight='bold')
        
        # 長條圖 (對數尺度)
        ax2.bar(labels, sizes, color=colors)
        ax2.set_yscale('log')
        ax2.set_ylabel('樣本數量 (對數尺度)')
        ax2.set_title('CTR 資料類別分布 (對數尺度)', fontsize=14, fontweight='bold')
        
        # 在長條上加上數值
        for i, v in enumerate(sizes):
            ax2.text(i, v * 1.1, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/01_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 模型參數設定
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'verbose': 10,  # 增加輸出以觀察訓練過程
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'min_data_in_leaf': 50,
            'scale_pos_weight': 1,
            'n_jobs': -1,
            'seed': 42,
            'num_boost_round': 2000,  # 增加最大迭代次數
            'reg_alpha': 0.1,  # 加入正則化
            'reg_lambda': 0.1,
        }
        
        # 3. 切分資料
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_ctr, test_size=0.2, random_state=42, stratify=self.y_ctr
        )
        
        # 4. 資料採樣與視覺化
        use_sampling = True
        if use_sampling and pos_count / len(self.y_ctr) < 0.01:
            print("執行資料採樣，平衡正負樣本比例...")
            pos_indices = np.where(y_tr == 1)[0]
            neg_indices = np.where(y_tr == 0)[0]
            
            target_ratio = 1
            sampled_neg_indices = np.random.choice(
                neg_indices, 
                size=min(len(neg_indices), len(pos_indices) * target_ratio), 
                replace=False
            )
            
            sampled_indices = np.concatenate([pos_indices, sampled_neg_indices])
            X_tr_sampled = X_tr.iloc[sampled_indices]
            y_tr_sampled = y_tr.iloc[sampled_indices]
            
            # 📊 視覺化2: 採樣前後對比
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 採樣前
            before_counts = [sum(y_tr == 0), sum(y_tr == 1)]
            axes[0].bar(['未點擊', '已點擊'], before_counts, color=['lightcoral', 'lightblue'])
            axes[0].set_title('採樣前', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('樣本數量')
            for i, v in enumerate(before_counts):
                axes[0].text(i, v * 1.02, f'{v:,}', ha='center', fontweight='bold')
            
            # 採樣後
            after_counts = [sum(y_tr_sampled == 0), sum(y_tr_sampled == 1)]
            axes[1].bar(['未點擊', '已點擊'], after_counts, color=['lightcoral', 'lightblue'])
            axes[1].set_title('採樣後', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('樣本數量')
            for i, v in enumerate(after_counts):
                axes[1].text(i, v * 1.02, f'{v:,}', ha='center', fontweight='bold')
            
            plt.suptitle('資料採樣前後對比', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/02_sampling_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            train_data = lgb.Dataset(X_tr_sampled, label=y_tr_sampled)
            print(f"採樣後訓練集大小: {len(X_tr_sampled)}, 正樣本比例: {sum(y_tr_sampled)/len(y_tr_sampled)*100:.2f}%")
        else:
            train_data = lgb.Dataset(X_tr, label=y_tr)

        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 5. 訓練模型 (儲存訓練歷史)
        print("開始模型訓練...")
        evals_result = {}
        self.ctr_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200), 
                lgb.log_evaluation(100),
                lgb.record_evaluation(evals_result)  # 記錄訓練過程
            ]
        )
        
        # 📊 視覺化3: 訓練過程曲線
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['binary_logloss', 'auc']
        titles = ['Binary Log Loss', 'AUC']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if i >= 2:  # 只繪製前兩個圖
                break
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            train_metric = evals_result['train'][metric]
            valid_metric = evals_result['valid'][metric]
            
            ax.plot(train_metric, label='Train', color='blue', alpha=0.7)
            ax.plot(valid_metric, label='Validation', color='red', alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(f'{title} 訓練曲線', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 第四個子圖：訓練資訊摘要
        axes[1, 1].axis('off')
        info_text = f"""
        訓練資訊摘要：
        
        • 總迭代次數: {self.ctr_model.best_iteration}
        • 早停輪次: 50
        • 學習率: {params['learning_rate']}
        • 樹的數量: {params['num_leaves']}
        • 正樣本權重: {params['scale_pos_weight']:.2f}
        
        • 訓練集大小: {len(train_data.get_label()):,}
        • 驗證集大小: {len(y_val):,}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, va='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/03_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 特徵重要性分析與視覺化
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.ctr_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print("\n前10個最重要特徵:")
        print(feature_importance.head(10))
        
        # 📊 視覺化4: 特徵重要性
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特徵重要性分數')
        plt.title('前15個最重要特徵', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 在長條上加上數值
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(importance + max(top_features['importance']) * 0.01, i, 
                    f'{importance:.0f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/04_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. 重要特徵篩選
        important_threshold = 10
        self.important_features = feature_importance[
            feature_importance['importance'] > important_threshold
        ]['feature'].tolist()
        
        print(f"\n重要特徵 (重要性 > {important_threshold}) 數量: {len(self.important_features)}")
        
        # 8. 模型評估與視覺化
        y_pred_val = self.ctr_model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred_val)
        ap = average_precision_score(y_val, y_pred_val)
        
        # 使用更適合不平衡資料的評估方式
        from sklearn.metrics import classification_report, f1_score

        # 設定更合理的決策閾值
        threshold = 0.1  # 而不是預設的 0.5
        y_pred_binary = (y_pred_val > threshold).astype(int)

        f1 = f1_score(y_val, y_pred_binary)
        print(f"F1 Score: {f1:.4f}")
        
        print(f"\n驗證集評估指標:")
        print(f"- AUC: {auc:.4f}")
        print(f"- Average Precision: {ap:.4f}")
        
        # 📊 視覺化5: ROC 曲線與 PR 曲線
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC 曲線
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC 曲線', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR 曲線
        precision, recall, _ = precision_recall_curve(y_val, y_pred_val)
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall 曲線', fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/05_roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 📊 視覺化6: 預測分布直方圖
        plt.figure(figsize=(12, 6))
        
        # 分別繪製正負樣本的預測分布
        y_pred_pos = y_pred_val[y_val == 1]
        y_pred_neg = y_pred_val[y_val == 0]
        
        plt.hist(y_pred_neg, bins=50, alpha=0.7, label=f'未點擊 (n={len(y_pred_neg)})', 
                 color='lightcoral', density=True)
        plt.hist(y_pred_pos, bins=50, alpha=0.7, label=f'已點擊 (n={len(y_pred_pos)})', 
                 color='lightblue', density=True)
        
        plt.xlabel('預測機率')
        plt.ylabel('密度')
        plt.title('CTR 預測機率分布', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 加上統計資訊
        plt.axvline(y_pred_neg.mean(), color='red', linestyle='--', alpha=0.8, 
                    label=f'未點擊平均: {y_pred_neg.mean():.6f}')
        plt.axvline(y_pred_pos.mean(), color='blue', linestyle='--', alpha=0.8, 
                    label=f'已點擊平均: {y_pred_pos.mean():.6f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/06_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 💾 儲存模型訓練報告
        report = f"""
        CTR 模型訓練報告
        ================
        
        資料統計：
        - 總樣本數: {len(self.y_ctr):,}
        - 正樣本數: {pos_count:,} ({pos_count/len(self.y_ctr)*100:.2f}%)
        - 負樣本數: {neg_count:,} ({neg_count/len(self.y_ctr)*100:.2f}%)
        - 不平衡比率: {neg_count/pos_count:.1f}:1
        
        模型參數：
        - 學習率: {params['learning_rate']}
        - 樹葉節點數: {params['num_leaves']}
        - 正樣本權重: {params['scale_pos_weight']:.2f}
        - 總迭代次數: {self.ctr_model.best_iteration}
        
        特徵統計：
        - 總特徵數: {len(self.feature_columns)}
        - 重要特徵數: {len(self.important_features)}
        
        評估指標：
        - AUC: {auc:.4f}
        - Average Precision: {ap:.4f}
        
        前5個重要特徵：
        {feature_importance.head(5).to_string(index=False)}
        
        視覺化檔案已儲存至: {viz_dir}/
        """
        
        with open(f'{viz_dir}/training_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ CTR 模型訓練完成!")
        print(f"📊 視覺化檔案已儲存至: {viz_dir}/")
        print(f"📄 訓練報告已儲存: {viz_dir}/training_report.txt")
        
        # 重要特徵重訓練邏輯保持不變...
        # (省略以節省空間，邏輯與原版相同)


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
        """預測單筆資料的勝價 - 極度保守版本"""
        try:
            if self.winprice_model is None:
                floor_price = row_series.get('ad_slot_floor_price', 1)
                return max(int(floor_price * 1.05), 1)  # 只比底價高5%
            
            df_row = pd.DataFrame([row_series])
            processed_df_row = self.preprocess_features(df_row, is_train=False)
            features_for_prediction = processed_df_row[self.feature_columns_winprice].fillna(0)
            
            try:
                log_duration = self.winprice_model.predict_expectation(features_for_prediction.iloc[0]).iloc[0]
                raw_pred = np.exp(log_duration)
            except:
                raw_pred = 15
            
            # 根據分析結果：你需要降低到原來的 1/30
            floor_price = row_series.get('ad_slot_floor_price', 1)
            
            # 極度保守調整
            conservative_pred = max(
                raw_pred * 0.03,     # 預測值的3% (原來高估30倍)
                floor_price * 1.02,  # 或底價的102%
                1
            )
            
            # 最高限制：絕對不超過8元
            return min(int(conservative_pred), 8)
            
        except Exception as e:
            floor_price = row_series.get('ad_slot_floor_price', 1)
            return max(int(floor_price), 1)


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
        if self.test_day2 is None:
            print("錯誤：測試資料未載入。")
            return None
        if self.ctr_model is None or self.winprice_model is None:
            print("錯誤：一個或多個模型未訓練。無法執行出價。")
            return None

        remaining_budget = self.DAY_BUDGET
        spent_per_hour = [0] * 24
        bid_results = []
        
        # 預處理整個測試集
        print("預處理整個測試集進行出價...")
        processed_test_df = self.preprocess_features(self.test_day2.copy(), is_train=False)
        print("測試集預處理完成。")

        # 批量預測CTR
        batch_size = 10000
        for i in range(0, len(processed_test_df), batch_size):
            batch = processed_test_df.iloc[i:i+batch_size]
            ctr_preds = self.ctr_model.predict(
                batch[self.feature_columns], 
                num_iteration=self.ctr_model.best_iteration
            )
            processed_test_df.loc[batch.index, 'predicted_ctr'] = ctr_preds
        
        # 檢查CTR預測結果
        print(f"CTR預測結果統計:")
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
        print("Win-Price 預測用特徵 NaN 檢查：")
        print(processed_test_df[self.winprice_features].isnull().sum())
        print("Win-Price 預測用特徵型態：")
        print(processed_test_df[self.winprice_features].dtypes)
        
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

        # 儲存結果
        result_df = pd.DataFrame(bid_results)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 新增這行
        output_filename = f"{self.student_id}_day1_{now_str}.csv"    # 修改這行
        result_df.to_csv(output_filename, index=False)
        print(f"\nDay1 出價完成，結果已儲存至: {output_filename}")
        print(f"總出價次數 (paying_price > 0): {(result_df['paying_price'] > 0).sum()}")
        print(f"總出價金額 (假設都贏得且按此價格支付): {result_df['paying_price'].sum()}")
        print(f"剩餘總預算: {remaining_budget:.2f}")
        
        return result_df

    def bid_day2(self):
        """執行 Day2 的出價邏輯 - 加入保守微調"""
        print("執行 Day2 出價...")
        if self.test_day2 is None:
            print("錯誤：測試資料未載入。")
            return None
        if self.ctr_model is None or self.winprice_model is None:
            print("錯誤：一個或多個模型未訓練。無法執行出價。")
            return None

        remaining_budget = self.DAY_BUDGET
        spent_per_hour = [0] * 24
        bid_results = []
        
        # 預處理整個測試集
        print("預處理整個測試集進行出價...")
        processed_test_df = self.preprocess_features(self.test_day2.copy(), is_train=False)
        print("測試集預處理完成。")

        # 批量預測CTR
        batch_size = 10000
        for i in range(0, len(processed_test_df), batch_size):
            batch = processed_test_df.iloc[i:i+batch_size]
            ctr_preds = self.ctr_model.predict(
                batch[self.feature_columns], 
                num_iteration=self.ctr_model.best_iteration
            )
            processed_test_df.loc[batch.index, 'predicted_ctr'] = ctr_preds
        
        # 檢查CTR預測結果
        print(f"CTR預測結果統計:")
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
        print("Win-Price 預測用特徵 NaN 檢查：")
        print(processed_test_df[self.winprice_features].isnull().sum())
        print("Win-Price 預測用特徵型態：")
        print(processed_test_df[self.winprice_features].dtypes)
        
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

                    # ========== 新增：保守出價策略 (來自 bid_day2) ==========
                    floor_price = original_row.get('ad_slot_floor_price', 1)
                    pctr = predicted_ctr
                    
                    # 三種策略取最小值
                    strategy1 = int(predicted_win_price * 0.6)  # 預測值60%
                    strategy2 = int(floor_price * 1.01)         # 底價101%
                    strategy3 = max(1, int(pctr * 100000))      # 基於CTR的出價
                    potential_bid = min(strategy1, strategy2, strategy3, 5)  # 最高5元
                    
                    # 額外性價比檢查
                    if potential_bid > 0:
                        rho = pctr / max(potential_bid, 1)
                        if rho < 2e-4:
                            potential_bid = 0
                    # ========== 保守出價策略結束 ==========
                    
                    # 確保最小出價
                    if potential_bid > 0:
                        potential_bid = max(potential_bid, 1)
                    
                    # 檢查預算限制
                    if potential_bid > 0 and remaining_budget >= potential_bid and \
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

        # 儲存結果
        result_df = pd.DataFrame(bid_results)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.student_id}_day1_{now_str}.csv"
        result_df.to_csv(output_filename, index=False)
        
        # 增強統計資訊
        final_bid_count = (result_df['paying_price'] > 0).sum()
        final_total = result_df['paying_price'].sum()
        avg_bid = result_df[result_df['paying_price'] > 0]['paying_price'].mean() if final_bid_count > 0 else 0
        
        print(f"\nDay1 出價完成，結果已儲存至: {output_filename}")
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
        rtb_system.bid_day2()
        
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