# src/config.py
import os

# --- Project Directories ---
# BASE_DIR sẽ là thư mục gốc của project (ví dụ: fraud_ml_ops_project/)
# Giả sử file này nằm trong src/, thì BASE_DIR là thư mục cha của src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns") # Thư mục cho MLflow log cục bộ

# --- Data File Paths ---
# Cập nhật tên file nếu cần
TRAIN_TRANSACTION_PATH = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY_PATH = os.path.join(RAW_DATA_DIR, "train_identity.csv")
# TEST_TRANSACTION_PATH = os.path.join(RAW_DATA_DIR, "test_transaction.csv") # File test từ cuộc thi (nếu dùng)
# TEST_IDENTITY_PATH = os.path.join(RAW_DATA_DIR, "test_identity.csv")     # File test từ cuộc thi (nếu dùng)

# --- Feature Engineering & Model Configs ---
TARGET_COL = 'isFraud'
TRANSACTION_DT_ORIGIN = '2017-11-30' # Kiểm tra lại ngày gốc này từ notebook của bạn
SECONDS_IN_DAY = 60 * 60 * 24

V_COLS_PREFIX = 'V'
V_COLS_RANGE = range(1, 340) # Từ V1 đến V339
PCA_N_COMPONENTS_V = 2 # Số chiều PCA cho V_cols
V_COLS_CORR_THRESHOLD = 0.8 # Ngưỡng tương quan cho V_cols. Đặt là None nếu không muốn chọn lọc V_cols.

M_COLS_FLAGS = [f'M{i}' for i in range(1, 10)] # M1 đến M9
C_COLS_FEATURES = [f'C{i}' for i in range(1, 15)] # C1 đến C14

# --- Model Parameters ---
LGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_estimators': 500, # Sẽ được sử dụng bởi lgb.train nếu num_boost_round không được truyền riêng
    'max_depth': 7,
    'seed': 42, # Để có thể tái tạo kết quả
    'n_jobs': -1, # Sử dụng tất cả các core có sẵn
    'verbose': -1, # Tắt bớt log của LightGBM (ngoại trừ log_evaluation)
    'mlflow_experiment_name': "Fraud_Detection_Local_Runs" # Tên experiment cho MLflow cục bộ
}
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_LOG_EVALUATION_PERIOD = 50

# --- Evaluation ---
PREDICTION_THRESHOLD = 0.5

# --- File Names for Saved Artifacts (Cục bộ) ---
MODEL_FILENAME = "lgbm_model.joblib"
MODEL_INFO_FILENAME = "model_info.json"
PCA_V_FILENAME = "pca_v_transformer.joblib" # Object PCA cho V_cols
SCALER_V_FILENAME = "scaler_v_transformer.joblib" # Object StandardScaler cho V_cols (nếu dùng trước PCA)
LABEL_ENCODERS_FILENAME = "label_encoders.joblib" # Dictionary các LabelEncoders đã fit
FINAL_TRAINING_FEATURES_FILENAME = "final_training_features.json" # Danh sách các feature cuối cùng dùng để train

# --- Azure ML Configuration (Dùng bởi run_pipeline.py và có thể cả train.py nếu muốn log trực tiếp) ---
# Các tên này sẽ được dùng khi tương tác với Azure ML
AZUREML_EXPERIMENT_NAME = "Fraud_Detection_AzureML_Pipeline" # Tên experiment trên Azure ML
AZUREML_REGISTERED_MODEL_NAME = "fraud-detection-lgbm-aml"
AZUREML_DEPLOYMENT_PACKAGE_NAME = "fraud-detection-deploy-pkg"  # Tên model sẽ đăng ký trên Azure ML Model Registry
# Các thông tin kết nối Azure ML Workspace sẽ được lấy từ file config.json ở thư mục gốc project
# hoặc từ biến môi trường Azure CLI đã đăng nhập.