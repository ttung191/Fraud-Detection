# src/predict.py
# ... (các import và phần tải model/features giữ nguyên như phiên bản đầy đủ trước) ...
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import json
# src/predict.py
import logging # Thêm import này
import os     # Đã có
from opencensus.ext.azure.log_exporter import AzureLogHandler # Thêm import này

from . import config
from . import data_preprocessing
from . import feature_engineering

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SRC_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# === BEGIN: CẤU HÌNH APPLICATION INSIGHTS LOGGING ===
# Lấy Connection String từ biến môi trường (Azure App Service sẽ tự động đặt biến này khi Application Insights được bật)
APPINSIGHTS_CONNECTION_STRING = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')

if APPINSIGHTS_CONNECTION_STRING:
    # Cấu hình root logger để bắt tất cả log từ ứng dụng và các thư viện
    # Bạn có thể tạo một logger cụ thể cho ứng dụng của mình nếu muốn
    # logger = logging.getLogger(__name__) # Nếu muốn logger cụ thể cho module này
    logger = logging.getLogger('') # Lấy root logger để bắt cả print nếu cấu hình đúng
    logger.setLevel(logging.INFO) # Đặt mức log, ví dụ: INFO, DEBUG

    # Tạo AzureLogHandler
    azure_handler = AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING)
    azure_handler.setLevel(logging.INFO) # Đặt mức log cho handler này

    # (Tùy chọn) Tạo một Formatter để định dạng log nếu muốn
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    azure_handler.setFormatter(formatter)

    # Thêm AzureLogHandler vào logger
    logger.addHandler(azure_handler)

    # (Quan trọng) Để print() cũng được gửi đi như log (mức INFO):
    # Điều này hoạt động bằng cách ghi đè sys.stdout, có thể không phải lúc nào cũng lý tưởng
    # nhưng tiện lợi cho việc bắt các print hiện có.
    # Cân nhắc sử dụng logging.info(), logging.error() thay vì print() trong code mới.
    import sys
    class StreamToLogger(object):
        """
        Fake file-like stream object that redirects writes to a logger instance.
        """
        def __init__(self, logger_instance, log_level=logging.INFO):
            self.logger = logger_instance
            self.log_level = log_level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass # sys.stdout không cần flush thực sự khi dùng logger

    # Chuyển hướng stdout (print) sang logger
    # Hãy cẩn thận với điều này trong môi trường production phức tạp
    # Nó có thể ảnh hưởng đến các thư viện khác cũng dùng stdout.
    # Nếu chỉ muốn log có chủ đích, hãy dùng logger.info(), logger.warning() trực tiếp.
    if not getattr(sys.stdout, 'is_a_logger_stream', False): # Tránh ghi đè nhiều lần
        sl = StreamToLogger(logger, logging.INFO)
        sl.is_a_logger_stream = True # Đánh dấu để không ghi đè lại
        sys.stdout = sl
        # (Tùy chọn) Chuyển hướng stderr
        # sl_err = StreamToLogger(logger, logging.ERROR)
        # sl_err.is_a_logger_stream = True
        # sys.stderr = sl_err
        logger.info("Stdout (print) is now redirected to Application Insights logger.")

    logger.info("Application Insights logging configured for the Flask app.")
    print("Test print: This message should go to Application Insights traces (via stdout redirection).")

else:
    # Tạo một logger cơ bản cho console nếu không có connection string (ví dụ khi chạy cục bộ không có AppInsights)
    logger = logging.getLogger(__name__) # Hoặc logging.getLogger('')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    if not logger.handlers: # Tránh thêm handler nhiều lần nếu code này được gọi lại
        logger.addHandler(console_handler)
    logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING not found. Using basic console logging.")
    print("Test print: This message goes to console (AppInsights not configured).")

# === END: CẤU HÌNH APPLICATION INSIGHTS LOGGING ===

MODEL_PATH = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)
TRAINING_FEATURES_PATH = os.path.join(config.ARTIFACTS_DIR, config.FINAL_TRAINING_FEATURES_FILENAME)

model = None
TRAINING_FEATURES = []
# ... (Logic tải model và TRAINING_FEATURES từ file như phiên bản trước đã cung cấp) ...
# Đảm bảo nó được tải đúng cách
try:
    if os.path.exists(MODEL_PATH): model = joblib.load(MODEL_PATH)
    if os.path.exists(TRAINING_FEATURES_PATH):
        with open(TRAINING_FEATURES_PATH, 'r') as f: TRAINING_FEATURES = json.load(f)
    elif model and hasattr(model, 'feature_name_'): TRAINING_FEATURES = model.feature_name_()
    if model: print(f"Model loaded. Expecting {len(TRAINING_FEATURES)} features.")
    else: print("ERROR: Model not loaded.")
except Exception as e:
    print(f"Error during model/feature_list loading: {e}")


def parse_form_value(form_value_str, target_type=str):
    if form_value_str is None or str(form_value_str).strip() == '':
        return np.nan
    try:
        if target_type == float: return float(form_value_str)
        if target_type == int: return int(float(form_value_str)) # float first for "123.0"
        if target_type == 'M_col': # For M columns (T, F, or empty for NaN)
            val_upper = str(form_value_str).upper()
            if val_upper == 'T': return 'T'
            if val_upper == 'F': return 'F'
            return np.nan 
        return str(form_value_str).strip() # Default to string
    except ValueError:
        return np.nan

def prepare_input_data_for_prediction(df_input_raw):
    # ... (Nội dung hàm này giữ nguyên như phiên bản đầy đủ trước) ...
    # Nó phải gọi:
    # 1. data_preprocessing.run_base_preprocessing(df, fit_encoders_flag=False)
    # 2. feature_engineering.run_all_feature_engineering(df, fit_transformers_flag=False, df_for_corr_fitting=None)
    # 3. df.fillna(-999)
    # 4. Đảm bảo đúng các features theo TRAINING_FEATURES (thêm cột thiếu, sắp xếp)
    if model is None: raise RuntimeError("Model is not loaded.")
    if not TRAINING_FEATURES: raise RuntimeError("Training features list not available.")

    df = df_input_raw.copy()
    print(f"  Raw input shape for preparation: {df.shape}")
    df = data_preprocessing.run_base_preprocessing(df, fit_encoders_flag=False)
    print(f"  Shape after base preprocessing: {df.shape}")
    df, _ = feature_engineering.run_all_feature_engineering(df, fit_transformers_flag=False, df_for_corr_fitting=None)
    print(f"  Shape after feature engineering: {df.shape}")
    df = df.fillna(-999) # Fill any NaNs that might have been generated
    print(f"  Shape after final fillna: {df.shape}")

    # Ensure all training features are present, fill with -999 if not
    for feature in TRAINING_FEATURES:
        if feature not in df.columns:
            print(f"    WARNING: Feature '{feature}' not in input. Adding and filling with -999.")
            df[feature] = -999
            
    try:
        df_final_features = df[TRAINING_FEATURES] # Select and reorder
    except KeyError as e:
        missing_cols = set(TRAINING_FEATURES) - set(df.columns)
        raise RuntimeError(f"Prediction data feature mismatch. Missing: {missing_cols}. Error: {e}")
    
    print(f"  Prediction data prepared. Shape: {df_final_features.shape}.")
    return df_final_features

@app.route("/api/predict", methods=["POST"])
def predict_api_endpoint():
    # ... (Logic API JSON giữ nguyên như phiên bản trước) ...
    if model is None: return jsonify({"error": "Model not available."}), 503
    try:
        data_json = request.get_json(force=True)
        if not data_json: return jsonify({"error": "No input data."}), 400
        df_input_raw = pd.DataFrame([data_json] if isinstance(data_json, dict) else data_json)
        X_prepared = prepare_input_data_for_prediction(df_input_raw)
        predictions_proba = model.predict(X_prepared, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else -1)
        predictions_binary = (predictions_proba >= config.PREDICTION_THRESHOLD).astype(int)
        if isinstance(data_json, dict):
            response = {"probability_fraud": float(predictions_proba[0]), "is_fraud": int(predictions_binary[0])}
        else:
            response = [{"probability_fraud": float(p), "is_fraud": int(b)} for p,b in zip(predictions_proba, predictions_binary)]
        return jsonify(response)
    except Exception as e:
        print(f"API ERROR: {str(e)}"); import traceback; traceback.print_exc()
        return jsonify({"error": f"API prediction error: {str(e)}"}), 500

@app.route("/", methods=["GET", "POST"])
def index_page():
    if model is None:
        return render_template("index.html", error_message="Error: Model not loaded. Please check server logs.")

    if request.method == "POST":
        try:
            raw_input_data = {}
            # Lấy giá trị từ form, bạn cần lặp qua TẤT CẢ các trường đã định nghĩa trong HTML
            
            # Core Transaction
            raw_input_data['TransactionID'] = parse_form_value(request.form.get('TransactionID'), target_type=int)
            raw_input_data['TransactionDT'] = parse_form_value(request.form.get('TransactionDT'), target_type=int) # Sẽ được chuyển sang datetime sau
            raw_input_data['TransactionAmt'] = parse_form_value(request.form.get('TransactionAmt'), target_type=float)
            raw_input_data['ProductCD'] = parse_form_value(request.form.get('ProductCD'))

            # Card Info
            for i in range(1, 7): # card1 to card6
                field_name = f'card{i}'
                target_type = float if field_name in ['card2', 'card3', 'card5'] else (int if field_name == 'card1' else str)
                raw_input_data[field_name] = parse_form_value(request.form.get(field_name), target_type=target_type)
            
            # Address & Email
            raw_input_data['addr1'] = parse_form_value(request.form.get('addr1'), target_type=float)
            raw_input_data['addr2'] = parse_form_value(request.form.get('addr2'), target_type=float)
            raw_input_data['dist1'] = parse_form_value(request.form.get('dist1'), target_type=float)
            raw_input_data['dist2'] = parse_form_value(request.form.get('dist2'), target_type=float)
            raw_input_data['P_emaildomain'] = parse_form_value(request.form.get('P_emaildomain'))
            raw_input_data['R_emaildomain'] = parse_form_value(request.form.get('R_emaildomain'))

            # C Columns
            for i in range(1, 15): raw_input_data[f'C{i}'] = parse_form_value(request.form.get(f'C{i}'), target_type=float)
            # D Columns
            for i in range(1, 16): raw_input_data[f'D{i}'] = parse_form_value(request.form.get(f'D{i}'), target_type=float)
            # M Columns
            for i in range(1, 10): raw_input_data[f'M{i}'] = parse_form_value(request.form.get(f'M{i}'), target_type='M_col') # 'M_col' sẽ được map_m_features_binary xử lý

            # id_ Columns - Cần xác định kiểu dữ liệu cẩn thận cho từng cột id_
            numerical_ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,32] # Cập nhật ds này
            string_ids = [12,15,16,23,27,28,29,30,31,33,34] # id_23 có thể là string (IP_PROXY...) hoặc số (nếu đã encode), id_27,28,29 cũng vậy
            m_flag_ids = [35,36,37,38]

            for i in numerical_ids: raw_input_data[f'id_{i:02d}'] = parse_form_value(request.form.get(f'id_{i:02d}'), target_type=float)
            for i in string_ids: raw_input_data[f'id_{i:02d}'] = parse_form_value(request.form.get(f'id_{i:02d}'))
            for i in m_flag_ids: raw_input_data[f'id_{i:02d}'] = parse_form_value(request.form.get(f'id_{i:02d}'), target_type='M_col')

            # Device Info
            raw_input_data['DeviceType'] = parse_form_value(request.form.get('DeviceType'))
            raw_input_data['DeviceInfo'] = parse_form_value(request.form.get('DeviceInfo'))

            # V Columns (V1-V339)
            for i in range(1, 340):
                raw_input_data[f'V{i}'] = parse_form_value(request.form.get(f'V{i}'), target_type=float)
            
            # Tạo DataFrame từ dữ liệu thô đã thu thập
            df_input_raw = pd.DataFrame([raw_input_data])
            print(f"WEB UI: Received form data, created DataFrame with shape {df_input_raw.shape}")
            
            X_prepared = prepare_input_data_for_prediction(df_input_raw)
            
            predictions_proba = model.predict(X_prepared, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else -1)
            predictions_binary = (predictions_proba >= config.PREDICTION_THRESHOLD).astype(int)

            result = {
                "probability_fraud": float(predictions_proba[0]),
                "is_fraud": int(predictions_binary[0])
            }
            return render_template("index.html", prediction_result=result, form_data=request.form) # Truyền lại form_data để giữ giá trị

        except Exception as e:
            import traceback
            print(f"WEB UI ERROR: {str(e)}")
            traceback.print_exc()
            return render_template("index.html", error_message=f"Processing error: {str(e)}", form_data=request.form)

    # GET request
    return render_template("index.html", prediction_result=None, error_message=None, form_data={})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)