# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from . import utils
from . import config

def load_and_merge_data(transaction_path, identity_path):
    print(f"Loading transaction data from: {transaction_path}")
    transaction_df = pd.read_csv(transaction_path)
    transaction_df = utils.reduce_memory_usage(transaction_df)

    print(f"Loading identity data from: {identity_path}")
    identity_df = pd.read_csv(identity_path)
    identity_df = utils.reduce_memory_usage(identity_df)

    print("Merging transaction and identity data...")
    df = pd.merge(transaction_df, identity_df, on='TransactionID', how='left')
    del transaction_df, identity_df # Free up memory
    return df

def preprocess_datetime(df):
    print("Preprocessing datetime features...")
    df['TransactionDT_Orig'] = df['TransactionDT'].copy() # Keep original for sorting if needed
    df['TransactionDT'] = pd.to_datetime(df['TransactionDT'], unit='s', origin=config.TRANSACTION_DT_ORIGIN)
    
    df['hour'] = df['TransactionDT'].dt.hour.astype(np.int8)
    df['day'] = df['TransactionDT'].dt.day.astype(np.int8)
    df['weekday'] = df['TransactionDT'].dt.weekday.astype(np.int8)
    df['TransactionDT_numeric'] = df['TransactionDT'].astype('int64') // 10**9 # Unix timestamp in seconds
    return df

def map_m_features_binary(df):
    print("Mapping M features to binary (T=1, F=0, NaN/Missing=-1)...")
    for col in config.M_COLS_FLAGS:
        if col in df.columns:
            df[col] = df[col].map({'T': 1, 'F': 0})
            df[col] = df[col].fillna(-1).astype(np.int8) # Consistent NaN handling
    return df

def clean_id_string_features(df):
    print("Cleaning id_30 and id_31 features...")
    # Use regex that captures version numbers too, then take the first part if needed.
    if 'id_30' in df.columns:
        df['id_30'] = df['id_30'].astype(str).str.extract(r'([a-zA-Z0-9\s\.]+)', expand=False).str.strip().fillna('missing_id30')
    if 'id_31' in df.columns:
        df['id_31'] = df['id_31'].astype(str).str.lower().str.split().str[0].fillna('missing_id31')
    return df

# src/data_preprocessing.py
# ... (các import và hàm khác) ...

def label_encode_categorical_features(df, fit_encoders=False, encoders_save_path=None, encoders_load_path=None):
    # ... (toàn bộ nội dung của hàm này như tôi đã cung cấp ở câu trả lời "Làm lại hết cho tôi") ...
    print(f"Label encoding categorical features (fit_encoders={fit_encoders})...")
    loaded_encoders = {}
    
    if not fit_encoders: # Load mode for inference/test
        if encoders_load_path and os.path.exists(encoders_load_path):
            loaded_encoders = joblib.load(encoders_load_path)
            print(f"Loaded label encoders from {encoders_load_path}")
        else:
            print(f"CRITICAL WARNING: fit_encoders is False, but encoder file not found at {encoders_load_path}. Cannot proceed with consistent label encoding.")
            return df, {} # Or raise error

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if config.TARGET_COL in categorical_cols:
        categorical_cols.remove(config.TARGET_COL)

    encoders_to_save = {} # Only used if fit_encoders is True

    for col in categorical_cols:
        nan_placeholder = 'NaN_SPECIAL_ENCODING_VALUE' 
        df[col] = df[col].astype(str).fillna(nan_placeholder)

        if fit_encoders:
            le = LabelEncoder()
            # Nếu đây là lần fit đầu tiên cho cột này, hoặc nếu loaded_encoders không chứa nó
            if col not in loaded_encoders: # Đảm bảo chỉ fit nếu encoder chưa tồn tại
                df[col] = le.fit_transform(df[col])
                encoders_to_save[col] = le
            else: # Nếu encoder đã tồn tại (ví dụ từ lần chạy trước cho cột khác, rồi giờ đến M1_M2_combo)
                  # thì không nên fit lại mà nên dùng cái đã có nếu logic là dùng chung file encoders.
                  # Tuy nhiên, logic hiện tại là tạo mới nếu fit_encoders=True.
                  # Nếu bạn muốn update file encoders, logic sẽ phức tạp hơn.
                  # Hiện tại, nếu fit_encoders=True, nó sẽ ghi đè encoder cũ của cột này nếu có.
                df[col] = le.fit_transform(df[col])
                encoders_to_save[col] = le

        else: # Transform mode
            if col in loaded_encoders:
                le = loaded_encoders[col]
                known_classes = set(le.classes_)
                
                unknown_val_encoded = -1 
                if nan_placeholder in known_classes:
                    unknown_val_encoded = le.transform([nan_placeholder])[0]
                else: 
                    print(f"WARNING: Placeholder '{nan_placeholder}' for NAs in column '{col}' was not in learned classes. New values will be mapped to {unknown_val_encoded}.")

                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_classes else unknown_val_encoded)
            else:
                print(f"WARNING: No pre-fitted encoder found for column '{col}' during transform. Filling with -1.")
                df[col] = -1 
    
    if fit_encoders and encoders_save_path:
        # Nếu encoders_to_save không rỗng (tức là có cột mới được fit)
        # hoặc nếu bạn muốn ghi đè toàn bộ file với các encoder đã được load và các encoder mới
        # Logic hiện tại là lưu tất cả các encoder đã được fit trong lần gọi này.
        # Nếu gọi lần 2 với fit_encoders=True cho M1_M2_combo, nó sẽ chỉ lưu encoder của M1_M2_combo
        # nếu encoders_to_save chỉ chứa nó.
        # Để cập nhật file encoders.joblib, bạn cần load file cũ, thêm encoder mới, rồi save lại.
        
        final_encoders_to_save = loaded_encoders.copy() # Bắt đầu với những gì đã load (nếu có)
        final_encoders_to_save.update(encoders_to_save) # Cập nhật/thêm encoder mới fit

        if final_encoders_to_save: # Chỉ lưu nếu có gì đó để lưu
            os.makedirs(os.path.dirname(encoders_save_path), exist_ok=True)
            joblib.dump(final_encoders_to_save, encoders_save_path)
            print(f"Saved/Updated label encoders to {encoders_save_path}")
        return df, final_encoders_to_save
        
    return df, loaded_encoders
# ... (phần còn lại của file data_preprocessing.py) ...


def run_base_preprocessing(df, fit_encoders_flag=False):
    df_copy = df.copy()
    df_copy = preprocess_datetime(df_copy)
    df_copy = map_m_features_binary(df_copy)
    df_copy = clean_id_string_features(df_copy)
    
    le_path = os.path.join(config.MODELS_DIR, config.LABEL_ENCODERS_FILENAME)
    
    df_copy, _ = label_encode_categorical_features(df_copy, 
                                                   fit_encoders=fit_encoders_flag, 
                                                   encoders_save_path=le_path if fit_encoders_flag else None,
                                                   encoders_load_path=le_path if not fit_encoders_flag else None)
    return df_copy