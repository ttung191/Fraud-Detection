# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

from . import config
from . import data_preprocessing 

def create_d_normalized_features(df):
    print("Creating D-normalized features (D1n, D4n, D10n, D15n)...")
    if 'TransactionDT_numeric' not in df.columns:
        raise ValueError("Column 'TransactionDT_numeric' is required for D-normalized features.")
    for d_col_num in [1, 4, 10, 15]:
        d_col = f'D{d_col_num}'
        if d_col in df.columns:
            df[f'{d_col}n'] = np.floor(df['TransactionDT_numeric'] / config.SECONDS_IN_DAY) - df[d_col].fillna(0)
    return df

def encode_AG(df_to_agg, main_columns, uids_list_of_lists, aggregations, fillna_val=-999):
    print(f"  Running Aggregation for uids: {uids_list_of_lists} on columns: {main_columns}")
    for main_col in main_columns:
        if main_col not in df_to_agg.columns:
            print(f"    Warning: Main column {main_col} not found. Skipping aggregations for it.")
            continue
        for col_group_list in uids_list_of_lists:
            existing_group_cols = [gc for gc in col_group_list if gc in df_to_agg.columns]
            if len(existing_group_cols) != len(col_group_list):
                print(f"    Warning: Not all grouping columns {col_group_list} found. Using {existing_group_cols}. Skipping if empty.")
                if not existing_group_cols: continue
            
            for agg_type in aggregations:
                new_col_name = f'{main_col}__{"_".join(existing_group_cols)}__{agg_type}'
                # print(f"      Creating {new_col_name}")
                temp_grouped = df_to_agg.groupby(existing_group_cols)[main_col]
                
                if agg_type == 'nunique': # nunique on NaNs can be tricky
                    agg_result = temp_grouped.nunique()
                else:
                    agg_result = temp_grouped.agg(agg_type)
                
                temp_agg = agg_result.reset_index(name=new_col_name)
                df_to_agg = df_to_agg.merge(temp_agg, on=existing_group_cols, how='left')
                df_to_agg[new_col_name] = df_to_agg[new_col_name].fillna(fillna_val)
    return df_to_agg

def create_uid_features(df):
    print("Creating UID features (uid, uid2, uid3) and aggregations...")
    # Columns used to create UIDs. Ensure they are strings and NaNs are handled.
    # D1n is numeric, card1, addr1 etc. might be numeric after label encoding or originally.
    uid_base_cols_map = {
        'card1': 'card1_str_uid', 'addr1': 'addr1_str_uid', 'D1n': 'D1n_str_uid',
        'card3': 'card3_str_uid', 'card5': 'card5_str_uid', 'addr2': 'addr2_str_uid'
    }
    for orig_col, temp_col in uid_base_cols_map.items():
        if orig_col in df.columns:
            df[temp_col] = df[orig_col].astype(str).fillna('missing')
        else:
            df[temp_col] = 'missing' # Create if not exists to avoid error

    df['uid']  = df['card1_str_uid'] + '_' + df['addr1_str_uid'] + '_' + df['D1n_str_uid']
    df['uid2'] = df['uid'] + '_' + df['card3_str_uid'] + '_' + df['card5_str_uid']
    df['uid3'] = df['uid2'] + '_' + df['addr1_str_uid'] + '_' + df['addr2_str_uid']
    
    df = encode_AG(df, ['D4n', 'D10n', 'D15n'], [['uid']], ['mean', 'std'])
    df = encode_AG(df, ['TransactionAmt', 'dist1'], [['uid']], ['mean', 'std'])
    df = encode_AG(df, ['C13'], [['uid']], ['nunique'])

    for uid_col_name in ['uid', 'uid2', 'uid3']:
        if uid_col_name not in df.columns: continue
        
        group = df.groupby(uid_col_name)
        df[f'{uid_col_name}_trans_count'] = group['TransactionDT_Orig'].transform('count') # Use original DT for count
        df[f'{uid_col_name}_amt_mean'] = group['TransactionAmt'].transform('mean')
        df[f'{uid_col_name}_amt_std'] = group['TransactionAmt'].transform('std')
        df[f'{uid_col_name}_amt_max'] = group['TransactionAmt'].transform('max')
        df[f'{uid_col_name}_amt_min'] = group['TransactionAmt'].transform('min')
        df[f'{uid_col_name}_amt_over_mean'] = df['TransactionAmt'] / (df[f'{uid_col_name}_amt_mean'] + 1e-6)
        
        df_sorted = df.sort_values(by=[uid_col_name, 'TransactionDT_Orig']) # Sort for shift
        if 'D1n' in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted['D1n']):
            prev_d1n = df_sorted.groupby(uid_col_name)['D1n'].shift(1)
            # Align index before assigning back to original df
            df[f'{uid_col_name}_delta_time'] = (df_sorted['D1n'] - prev_d1n).reindex(df.index)
        else:
            df[f'{uid_col_name}_delta_time'] = np.nan
            
        df[f'{uid_col_name}_rolling_trans_6h'] = group['TransactionDT_Orig'].fillna(0).transform(
            lambda x: x.rolling(window=3, min_periods=1).count()
        )

    if 'P_emaildomain' in df.columns:
        df['email_suffix'] = df['P_emaildomain'].astype(str).str.extract(r'(@[^.]+)', expand=False).fillna('missing_email')
        df['is_free_email'] = df['email_suffix'].isin(['@gmail', '@yahoo', '@hotmail', '@outlook']).astype(int)
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo_clean'] = df['DeviceInfo'].astype(str).str.lower().str.extract(r'([a-zA-Z]+)', expand=False).fillna('unknown_device')
        df['is_known_device'] = df['DeviceInfo_clean'].isin(['samsung', 'huawei', 'apple', 'lenovo', 'windows', 'ios', 'sm', 'lg']).astype(int)

    df = df.drop(columns=[temp_col for temp_col in uid_base_cols_map.values() if temp_col in df.columns], errors='ignore')
    return df # fillna will be done at the end of all feature engineering

def create_c_features(df):
    print("Creating C-column features...")
    c_cols = [f'C{i}' for i in config.C_COLS_FEATURES if f'C{i}' in df.columns]
    if not c_cols: return df
    for c in c_cols: df[c] = df[c].fillna(0)
    df['C_sum'] = df[c_cols].sum(axis=1)
    df['C_mean'] = df[c_cols].mean(axis=1)
    df['C_std'] = df[c_cols].std(axis=1).fillna(0)
    for col in c_cols:
        df[f'{col}_over_mean'] = df[col] / (df['C_mean'] + 1e-6)
        df[f'{col}_is_outlier'] = (df[col] > (df['C_mean'] + 2 * df['C_std'])).astype(int)
    return df

def create_m_flag_aggregates(df):
    print("Creating M-flag aggregate features...")
    # M_cols_flags should already be 1/0/-1 from data_preprocessing
    m_cols_numeric = [col for col in config.M_COLS_FLAGS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not m_cols_numeric: return df
    
    df['M_flags_sum'] = df[m_cols_numeric].sum(axis=1)
    df['M_flags_mean'] = df[m_cols_numeric].mean(axis=1) # NaNs (if any -1 were NaNs) will be handled by mean
    
    if 'M1' in df.columns and 'M2' in df.columns:
        # M1, M2 are already numeric (1,0,-1). Convert to string for combo.
        df['M1_str_combo'] = df['M1'].astype(str) 
        df['M2_str_combo'] = df['M2'].astype(str)
        df['M1_M2_combo'] = df['M1_str_combo'] + '_' + df['M2_str_combo']
        df.drop(columns=['M1_str_combo', 'M2_str_combo'], inplace=True, errors='ignore')
    return df

def select_uncorrelated_v_features(df_for_fitting_corr, threshold=config.V_COLS_CORR_THRESHOLD):
    if threshold is None:
        print("V_COLS_CORR_THRESHOLD is None, skipping V_cols selection.")
        return [] # Return empty list if no selection
        
    print(f"Selecting uncorrelated V features with threshold {threshold}...")
    v_cols_original = [f'{config.V_COLS_PREFIX}{i}' for i in config.V_COLS_RANGE if f'{config.V_COLS_PREFIX}{i}' in df_for_fitting_corr.columns]
    if not v_cols_original:
        print("No V-columns (V1-V339) found for correlation selection.")
        return []

    correlation_matrix = df_for_fitting_corr[v_cols_original].corr().abs()
    selected_v_columns = []
    for col in correlation_matrix.columns:
        if not selected_v_columns:
            selected_v_columns.append(col)
        elif all(correlation_matrix[col][selected_v_columns].fillna(0) < threshold):
            selected_v_columns.append(col)
    print(f"Selected {len(selected_v_columns)} uncorrelated V columns (from V1-V339): {selected_v_columns[:5]}...")
    return selected_v_columns

def create_v_features_pca(df, fit_transformers=False, n_components=config.PCA_N_COMPONENTS_V):
    scaler_path=os.path.join(config.MODELS_DIR, config.SCALER_V_FILENAME)
    pca_path=os.path.join(config.MODELS_DIR, config.PCA_V_FILENAME)
    
    print(f"Creating V-column features (PCA, mean, std) (fit_transformers={fit_transformers})...")
    v_cols = [f'{config.V_COLS_PREFIX}{i}' for i in config.V_COLS_RANGE if f'{config.V_COLS_PREFIX}{i}' in df.columns]
    
    df_v_fillna = df[v_cols].fillna(-999) if v_cols else pd.DataFrame()

    if df_v_fillna.empty:
        print("No V-columns found. Creating default V_PCA, V_mean, V_std features.")
        for i in range(n_components): df[f'V_PCA_{i+1}'] = -999
        df['V_mean'] = -999
        df['V_std'] = 0
        return df

    v_scaled = None
    if fit_transformers:
        scaler = StandardScaler()
        v_scaled = scaler.fit_transform(df_v_fillna)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Saved StandardScaler for V_cols to {scaler_path}")

        pca_transformer = PCA(n_components=n_components, random_state=42)
        v_pca_transformed = pca_transformer.fit_transform(v_scaled)
        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        joblib.dump(pca_transformer, pca_path)
        print(f"Saved PCA transformer for V_cols to {pca_path}")
    else: # Load and transform
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            v_scaled = scaler.transform(df_v_fillna)
            print(f"Loaded and applied StandardScaler for V_cols from {scaler_path}")
        else:
            print(f"CRITICAL WARNING: StandardScaler for V_cols not found at {scaler_path}. Cannot proceed with PCA.")
            for i in range(n_components): df[f'V_PCA_{i+1}'] = -999 # Default values
            df['V_mean'] = df_v_fillna.mean(axis=1)
            df['V_std'] = df_v_fillna.std(axis=1).fillna(0)
            return df

        if os.path.exists(pca_path) and v_scaled is not None:
            pca_transformer = joblib.load(pca_path)
            v_pca_transformed = pca_transformer.transform(v_scaled)
            print(f"Loaded and applied PCA transformer for V_cols from {pca_path}")
        else:
            print(f"CRITICAL WARNING: PCA transformer for V_cols not found at {pca_path} or v_scaled is None. V_PCA features will be default.")
            for i in range(n_components): df[f'V_PCA_{i+1}'] = -999
            df['V_mean'] = df_v_fillna.mean(axis=1)
            df['V_std'] = df_v_fillna.std(axis=1).fillna(0)
            return df
            
    for i in range(n_components):
        df[f'V_PCA_{i+1}'] = v_pca_transformed[:, i]
    
    df['V_mean'] = df_v_fillna.mean(axis=1)
    df['V_std'] = df_v_fillna.std(axis=1).fillna(0)
    return df

def run_all_feature_engineering(df, fit_transformers_flag=False, df_for_corr_fitting=None):
    df_copy = df.copy()
    
    df_copy = create_d_normalized_features(df_copy)
    df_copy = create_uid_features(df_copy) # uid features depend on D1n
    df_copy = create_c_features(df_copy)
    df_copy = create_v_features_pca(df_copy, 
                                   fit_transformers=fit_transformers_flag)
    df_copy = create_m_flag_aggregates(df_copy)

    # Label encode M1_M2_combo if it was created and is object type
    if 'M1_M2_combo' in df_copy.columns and df_copy['M1_M2_combo'].dtype == 'object':
        le_path = os.path.join(config.MODELS_DIR, config.LABEL_ENCODERS_FILENAME)
        # This call will only affect M1_M2_combo as other categoricals should be done
        # It re-uses the label_encoders.joblib, so M1_M2_combo's encoder will be added/loaded
        df_copy, _ = data_preprocessing.label_encode_categorical_features(df_copy,
                                                                         fit_encoders=fit_transformers_flag,
                                                                         encoders_save_path=le_path if fit_transformers_flag else None,
                                                                         encoders_load_path=le_path if not fit_transformers_flag else None)


    selected_v_cols_list = []
    if fit_transformers_flag: # Only select uncorrelated V columns during training phase
        # Use df_for_corr_fitting (which should be the data *before* V_cols are heavily modified by PCA)
        # Typically, this would be df_processed from run_pipeline.py
        if df_for_corr_fitting is None: 
            print("WARNING: df_for_corr_fitting not provided for V-column correlation selection. Using current df which might be suboptimal.")
            df_for_corr_fitting = df_copy 
        selected_v_cols_list = select_uncorrelated_v_features(
            df_for_corr_fitting, 
            threshold=config.V_COLS_CORR_THRESHOLD
        )
    
    return df_copy, selected_v_cols_list