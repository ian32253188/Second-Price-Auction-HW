import os
import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 檢查並導入必要的函式庫 ---
try:
    from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
    from deepctr.models import DeepFM
    from tensorflow.keras.optimizers import Adam
    DEEPCTR_AVAILABLE = True
except ImportError:
    print("警告: deepctr 或 tensorflow 未安裝。請執行 'pip install deepctr tensorflow'")
    DEEPCTR_AVAILABLE = False

try:
    from lifelines import WeibullAFTFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    print("警告: lifelines 未安裝。請執行 'pip install lifelines'")
    LIFELINES_AVAILABLE = False

try:
    from user_agents import parse as ua_parse
    USER_AGENTS_AVAILABLE = True
except ImportError:
    print("警告: user_agents 未安裝。請執行 'pip install pyyaml ua-parser user-agents'")
    USER_AGENTS_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- 特徵工程輔助函式 ---

def parse_user_agent(ua_str):
    """解析 user_agent 字串，提取瀏覽器、作業系統和設備類型"""
    ua = ua_parse(str(ua_str))
    browser = ua.browser.family or "Unknown"
    os_family = ua.os.family or "Unknown"
    
    if ua.is_mobile:
        device_type = "Mobile"
    elif ua.is_tablet:
        device_type = "Tablet"
    elif ua.is_pc:
        device_type = "PC"
    else:
        device_type = "Other"
    return pd.Series([browser, os_family, device_type])

def parse_ad_slot_id(slot_id):
    """從 feature_test02.ipynb 來的 ad_slot_id 解析函式"""
    s = str(slot_id)
    result = {
        'slot_p1': None, 'slot_p2': None, 'slot_p3': None,
        'slot_platform': None, 'slot_platform_id1': None, 'slot_platform_id2': None,
        'slot_category': None, 'slot_flag': None, 'slot_width_tier': None,
        'slot_numeric_id': None, 'slot_unknown_id': None
    }
    if s.startswith('mm_'):
        parts = s.split('_')
        if len(parts) == 4:
            result['slot_p1'], result['slot_p2'], result['slot_p3'] = parts[1], parts[2], parts[3]
        else:
            result['slot_unknown_id'] = s
    elif any(s.startswith(prefix) for prefix in ['discuz_', 'phpwind_', 'dz_', 'pw_']):
        parts = s.split('_')
        result['slot_platform'] = parts[0]
        if len(parts) > 1: result['slot_platform_id1'] = parts[1]
        if len(parts) > 2: result['slot_platform_id2'] = parts[2]
    elif s.isdigit():
        result['slot_numeric_id'] = s
    elif '_' in s:
        parts = s.split('_')
        result['slot_category'] = parts[0]
        remaining_parts = parts[1:]
        temp_flag = []
        for part in remaining_parts:
            if part.startswith('Width'):
                result['slot_width_tier'] = part
            elif len(part) == 1 and part.isupper():
                temp_flag.append(part)
        if temp_flag:
            result['slot_flag'] = '_'.join(temp_flag)
    else:
        result['slot_unknown_id'] = s
    return pd.Series(result)

# --- 主要的 RTB 競價系統類別 ---

class RTBBiddingSystem:
    def __init__(self, student_id="M36134016"):
        """初始化系統參數、模型和資料處理器"""
        self.student_id = student_id
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 建立輸出資料夾
        self.output_dir = f"bidding_system_run_{self.timestamp}"
        self.ctr_viz_dir = os.path.join(self.output_dir, "ctr_model_visualization")
        self.wp_viz_dir = os.path.join(self.output_dir, "winprice_model_visualization")
        os.makedirs(self.ctr_viz_dir, exist_ok=True)
        os.makedirs(self.wp_viz_dir, exist_ok=True)

        # 模型
        self.ctr_model = None
        self.winprice_model = None
        
        # 資料處理器 (Encoders, Scalers, Vocab)
        self.encoders = {}
        self.scalers = {}
        self.user_tags_vocab = {'<PAD>': 0}
        self.user_tags_max_len = 0

        # DeepCTR 特徵欄位
        self.deepctr_feature_columns = []
        self.deepctr_feature_names = None
        
        # 競價策略超參數
        self.base_bid = 65  # 基礎出價，可調整
        self.avg_ctr = 0.0  # 平均 pCTR，將在訓練後計算
        self.DAY_BUDGET = 5000 # 每日總預算

    def _preprocess_features(self, df, is_train=True):
        """對 DataFrame 進行完整的特徵工程"""
        print(f"開始特徵預處理 (is_train={is_train})...")
        df_processed = df.copy()

        # 1. 捨棄無用欄位
        for col in ['anonymous_url_id', 'key_page_url', 'bid_id']:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)

        # 2. 時間特徵
        # 自動判斷 timestamp 欄位格式
        if np.issubdtype(df_processed['timestamp'].dtype, np.number):
            df_processed['timestamp_dt'] = pd.to_datetime(df_processed['timestamp'], unit='s')
        else:
            df_processed['timestamp_dt'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
        df_processed['hour'] = df_processed['timestamp_dt'].dt.hour
        df_processed['day_of_week'] = df_processed['timestamp_dt'].dt.dayofweek

        # 3. User Agent 解析
        print("   - 解析 User Agent...")
        ua_features = df_processed['user_agent'].apply(parse_user_agent)
        ua_features.columns = ['browser', 'os', 'device_type']
        df_processed = pd.concat([df_processed, ua_features], axis=1)

        # 4. Ad Slot ID 解析
        print("   - 解析 Ad Slot ID...")
        ad_slot_features = df_processed['ad_slot_id'].apply(parse_ad_slot_id)
        df_processed = pd.concat([df_processed, ad_slot_features], axis=1)

        # 5. 新特徵: ad_slot_size
        df_processed['ad_slot_size'] = df_processed['ad_slot_width'] * df_processed['ad_slot_height']

        # 6. User Tags (多熱點特徵)
        print("   - 處理 User Tags...")
        df_processed['user_tags'] = df_processed['user_tags'].fillna('').astype(str)
        tags_list = df_processed['user_tags'].apply(lambda x: x.split(',') if x else [])
        
        if is_train:
            # 建立詞典
            all_tags = [tag for sublist in tags_list for tag in sublist if tag]
            for tag in all_tags:
                if tag not in self.user_tags_vocab:
                    self.user_tags_vocab[tag] = len(self.user_tags_vocab)
            # 計算最大長度
            self.user_tags_max_len = max(len(x) for x in tags_list)
        
        # 將 tags 轉換為索引序列
        tags_seq = tags_list.apply(lambda t_list: [self.user_tags_vocab.get(t, 0) for t in t_list])
        df_processed['user_tags_seq'] = list(pad_sequences(tags_seq, maxlen=self.user_tags_max_len, padding='post'))

        # 7. 類別特徵與數值特徵定義
        categorical_features = [
            'domain', 'ip', 'city', 'region', 'ad_exchange', 'ad_slot_format',
            'ad_slot_visibility', 'creative_id', 'browser', 'os', 'device_type',
            'hour', 'day_of_week', 'slot_p1', 'slot_p2', 'slot_p3', 'slot_platform',
            'slot_platform_id1', 'slot_platform_id2', 'slot_category', 'slot_flag',
            'slot_width_tier', 'slot_numeric_id', 'slot_unknown_id'
        ]
        numerical_features = ['ad_slot_width', 'ad_slot_height', 'ad_slot_floor_price', 'ad_slot_size']

        # 8. 處理類別特徵 (Label Encoding)
        print("   - 進行標籤編碼...")
        for col in categorical_features:
            df_processed[col] = df_processed[col].fillna('NULL').astype(str)
            if is_train:
                self.encoders[col] = LabelEncoder().fit(df_processed[col])
            
            # 對於測試集，如果出現訓練集未見過的類別，給予一個預設值 (例如 -1 或 len(classes))
            known_classes = set(self.encoders[col].classes_)
            df_processed[col] = df_processed[col].apply(lambda x: x if x in known_classes else 'NULL')
            df_processed[col] = self.encoders[col].transform(df_processed[col])

        # 9. 處理數值特徵 (Min-Max Scaling)
        print("   - 進行數值縮放...")
        for col in numerical_features:
            df_processed[col] = df_processed[col].fillna(0)
            if is_train:
                self.scalers[col] = MinMaxScaler().fit(df_processed[[col]])
            df_processed[col] = self.scalers[col].transform(df_processed[[col]])
        
        print("特徵預處理完成。")
        return df_processed, categorical_features, numerical_features

    def train_ctr_model(self, train_df, cat_feats, num_feats):
        """訓練 DeepFM 模型來預測 pCTR"""
        print("--- 開始訓練 pCTR 模型 (DeepFM) ---")
        
        target = ['click']
        y_train = train_df[target]
        
        # 建立 DeepCTR 特徵欄位
        sparse_features = [SparseFeat(feat, vocabulary_size=len(self.encoders[feat].classes_), embedding_dim=8) for feat in cat_feats]
        dense_features = [DenseFeat(feat, 1) for feat in num_feats]
        varlen_features = [VarLenSparseFeat(SparseFeat('user_tags_seq', vocabulary_size=len(self.user_tags_vocab), embedding_dim=8), maxlen=self.user_tags_max_len, combiner='mean')]
        
        self.deepctr_feature_columns = sparse_features + dense_features + varlen_features
        self.deepctr_feature_names = get_feature_names(self.deepctr_feature_columns)

        # 準備模型輸入
        train_model_input = {name: train_df[name] for name in self.deepctr_feature_names if name != 'user_tags_seq'}
        train_model_input['user_tags_seq'] = np.vstack(train_df['user_tags_seq'].values)

        # 建立、編譯、訓練模型
        self.ctr_model = DeepFM(self.deepctr_feature_columns, self.deepctr_feature_columns, task='binary')
        self.ctr_model.compile(Adam(learning_rate=0.001), "binary_crossentropy", metrics=['binary_crossentropy', 'AUC'])
        
        history = self.ctr_model.fit(train_model_input, y_train, batch_size=2048, epochs=3, verbose=1, validation_split=0.2)
        
        # 計算並儲存 avgCTR
        all_pctr_preds = self.ctr_model.predict(train_model_input, batch_size=2048).flatten()
        self.avg_ctr = np.mean(all_pctr_preds)
        print(f"pCTR 模型訓練完成。平均 pCTR (avgCTR) = {self.avg_ctr:.6f}")

        # 儲存模型
        model_path = os.path.join(self.output_dir, "deepfm_model.h5")
        self.ctr_model.save(model_path)
        print(f"模型已儲存至 {model_path}")

        return history, y_train, all_pctr_preds

    def train_winprice_model(self, train_df, cat_feats, num_feats):
        """使用生存分析訓練得標價格模型"""
        print("--- 開始訓練得標價格模型 (WeibullAFT) ---")
        
        # 生存分析需要事件 (event) 和持續時間 (duration)
        # 在此案例中，所有訓練資料都是得標的，所以 event=1
        # duration 是得標價格 paying_price
        win_df = train_df.copy()
        win_df['event'] = 1
        win_df['duration'] = win_df['paying_price']

        features_for_winprice = cat_feats + num_feats
        
        self.winprice_model = WeibullAFTFitter()
        self.winprice_model.fit(win_df[features_for_winprice + ['duration', 'event']], 'duration', event_col='event')
        
        print("得標價格模型訓練完成。")
        self.winprice_model.print_summary()
        
        # 視覺化並儲存
        plt.figure(figsize=(12, 7))
        self.winprice_model.plot()
        plt.title("Weibull AFT Model - Parameter Coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(self.wp_viz_dir, "01_coefficients.png"))
        plt.close()

    def generate_visualizations(self, history, y_true, pctr_pred):
        """產生並儲存所有視覺化圖表"""
        print("--- 產生視覺化報告 ---")
        
        # 1. 訓練歷史
        pd.DataFrame(history.history).plot(figsize=(10, 6))
        plt.grid(True)
        plt.title('DeepFM Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.savefig(os.path.join(self.ctr_viz_dir, "01_training_history.png"))
        plt.close()

        # 2. ROC 曲線
        fpr, tpr, _ = roc_curve(y_true, pctr_pred)
        auc = roc_auc_score(y_true, pctr_pred)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(os.path.join(self.ctr_viz_dir, "02_roc_curve.png"))
        plt.close()

        # 3. pCTR 預測分佈
        plt.figure(figsize=(10, 6))
        sns.histplot(pctr_pred, bins=50, kde=True)
        plt.title('pCTR Prediction Distribution on Training Data')
        plt.xlabel('Predicted pCTR')
        plt.ylabel('Frequency')
        plt.axvline(self.avg_ctr, color='r', linestyle='--', label=f'Avg CTR: {self.avg_ctr:.4f}')
        plt.legend()
        plt.savefig(os.path.join(self.ctr_viz_dir, "03_prediction_distribution.png"))
        plt.close()
        
        print(f"視覺化圖表已儲存至 {self.ctr_viz_dir} 和 {self.wp_viz_dir}")

    def predict_and_bid(self, cat_feats, num_feats):
        """讀取測試資料，進行預測並產生提交檔案（包含預算控制）"""
        print("\n--- 開始處理測試資料並競價 ---")
        
        # 讀取測試資料
        try:
            test_df_raw = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\test_day3.csv')
            print(f"成功讀取 test_day3.csv，共 {len(test_df_raw)} 筆資料。")
        except FileNotFoundError:
            print("錯誤: 找不到 test_day3.csv。")
            return

        # 儲存 bid_id 以便最後合併
        bid_ids = test_df_raw['bid_id']

        # 預處理測試資料
        test_df, _, _ = self._preprocess_features(test_df_raw, is_train=False)
        
        # 準備模型輸入
        test_model_input = {name: test_df[name] for name in self.deepctr_feature_names if name != 'user_tags_seq'}
        test_model_input['user_tags_seq'] = np.vstack(test_df['user_tags_seq'].values)

        # 預測 pCTR
        print("   - 使用 DeepFM 預測 pCTR...")
        test_pctr = self.ctr_model.predict(test_model_input, batch_size=4096).flatten()

        # 應用線性出價策略 (初始出價)
        print("   - 應用線性出價策略...")
        initial_bids = self.base_bid * (test_pctr / self.avg_ctr)

        # --- 預算控制調整 ---
        print("   - 預測得標價格以進行預算調整...")
        features_for_winprice = cat_feats + num_feats
        predicted_win_prices = self.winprice_model.predict_median(test_df[features_for_winprice])

        # 估算總成本: 假設我們只對出價高於預測市場價的廣告進行投標，成本為預測市場價
        potential_wins_mask = initial_bids > predicted_win_prices
        estimated_total_cost = predicted_win_prices[potential_wins_mask].sum()

        print(f"   - 預估總花費 (未調整): {estimated_total_cost:.2f}")
        print(f"   - 每日總預算: {self.DAY_BUDGET}")

        # 計算預算調整因子
        if estimated_total_cost > 1: # 避免因過小的值導致因子過大
            budget_adjustment_factor = self.DAY_BUDGET / estimated_total_cost
            # 為了避免出價過高或過低，可以對調整因子設定上下限
            budget_adjustment_factor = np.clip(budget_adjustment_factor, 0.5, 1.5) 
            print(f"   - 預算調整因子: {budget_adjustment_factor:.4f}")
        else:
            # 如果預估花費為0或極小，可能代表出價都太低，使用一個預設值來提高出價
            budget_adjustment_factor = 1.2 
            print(f"   - 預估花費過低，使用預設調整因子 {budget_adjustment_factor}")

        # 應用調整因子得到最終出價
        final_bids = initial_bids * budget_adjustment_factor
        
        # 建立提交 DataFrame
        submission_df = pd.DataFrame({'bid_id': bid_ids, 'bidding_price': final_bids})
        
        # 儲存提交檔案
        submission_filename = f"{self.student_id}_day3.csv"
        submission_path = os.path.join(self.output_dir, submission_filename)
        submission_df.to_csv(submission_path, index=False)
        
        print(f"競價完成！提交檔案已儲存至: {submission_path}")

    def run(self):
        """執行完整的競價流程"""
        print(f"====== RTB 競價系統執行開始 ({self.timestamp}) ======")
        
        # 檢查函式庫是否可用
        if not all([DEEPCTR_AVAILABLE, LIFELINES_AVAILABLE, USER_AGENTS_AVAILABLE]):
            print("錯誤: 缺少必要的 Python 函式庫，請根據上方的警告訊息進行安裝。")
            return

        # 1. 載入並預處理訓練資料
        try:
            train_df_raw = pd.read_csv(r'C:\Users\ian32\Downloads\final\data\train.csv')
            print(f"成功讀取 train.csv，共 {len(train_df_raw)} 筆資料。")
        except FileNotFoundError:
            print("錯誤: 找不到 train.csv，無法繼續執行。")
            return
            
        train_df, cat_feats, num_feats = self._preprocess_features(train_df_raw, is_train=True)

        # 2. 訓練 pCTR 模型
        history, y_true, pctr_pred = self.train_ctr_model(train_df, cat_feats, num_feats)

        # 3. 訓練得標價格模型
        self.train_winprice_model(train_df, cat_feats, num_feats)

        # 4. 產生視覺化報告
        self.generate_visualizations(history, y_true, pctr_pred)

        # 5. 處理測試集並出價
        self.predict_and_bid(cat_feats, num_feats)
        
        print(f"\n====== RTB 競價系統執行完畢 ======")


if __name__ == "__main__":
    system = RTBBiddingSystem()
    system.run()
