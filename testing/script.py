from flask import Flask, request
import mlflow
import numpy as np
import pandas as pd
import joblib

#step1 mapping the columns names of CICflowmeter json to dataset columns names

rename_dict = {
    'dst_port': 'Destination Port',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Backward Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'totlen_bwd_pkts': 'Total Length of Bwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'fwd_iat_mean': 'Flow IAT Mean',
    'fwd_iat_std': 'Flow IAT Std',
    'fwd_iat_max': 'Flow IAT Max',
    'fwd_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'flow_iat_mean': 'Fwd IAT Mean',
    'flow_iat_std': 'Fwd IAT Std',
    'flow_iat_max': 'Fwd IAT Max',
    'flow_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'rst_flag_cnt': 'RST Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'urg_flag_cnt': 'URG Flag Count',
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    'fwd_seg_size_avg': 'Avg Fwd Segment Size',
    'bwd_seg_size_avg': 'Avg Bwd Segment Size',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'subflow_bwd_byts': 'Subflow Bwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'fwd_seg_size_min': 'min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min'
}

numerical_columns = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]
categorical_columns= ['Fwd PSH Flags',
 'Fwd URG Flags',
 'FIN Flag Count',
 'RST Flag Count',
 'PSH Flag Count',
 'ACK Flag Count',
 'URG Flag Count',
 'ECE Flag Count']

feature_selected_binary = ['Average Packet Size', 'Packet Length Variance', 'Packet Length Std', 'Destination Port', 'Packet Length Mean', 'Total Length of Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Max Packet Length', 'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Init_Win_bytes_backward', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Flow IAT Max', 'Flow Bytes/s', 'Flow Duration', 'Bwd Packets/s', 'Bwd Header Length', 'Fwd Packets/s', 'Fwd IAT Max', 'Flow Packets/s', 'Fwd Header Length', 'Flow IAT Mean', 'Bwd IAT Max', 'Fwd IAT Total', 'Bwd Packet Length Std', 'Bwd IAT Total', 'Fwd IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean', 'Flow IAT Std', 'Total Backward Packets', 'Fwd Packet Length Min', 'Min Packet Length', 'Bwd Packet Length Min', 'Fwd IAT Std', 'Active Min', 'Active Mean', 'Active Max']
feature_selected_multiclass = ['Packet Length Variance', 'Packet Length Std', 'Packet Length Mean', 'Average Packet Size', 'Total Length of Bwd Packets', 'Subflow Bwd Bytes', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Init_Win_bytes_backward', 'Destination Port', 'Max Packet Length', 'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Bwd Packet Length Std', 'Flow IAT Max', 'Init_Win_bytes_forward', 'Fwd IAT Max', 'Bwd Packets/s', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Flow Bytes/s', 'Fwd Packet Length Std', 'Fwd Header Length', 'Flow Packets/s', 'Flow IAT Mean', 'Flow Duration', 'Bwd Header Length', 'Fwd Packets/s', 'Fwd IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Std', 'Bwd IAT Max', 'Total Backward Packets', 'Bwd IAT Total', 'Bwd IAT Mean', 'Total Fwd Packets', 'Bwd IAT Std', 'Active Min', 'Active Mean', 'Idle Max', 'Active Max', 'Bwd IAT Min']

# ToDo : initialze the logging function
def rename_columns(incoming_df):#renaming yhe columns of the incoming dataframe
    incoming_df.rename(columns=rename_dict, inplace=True) # we change the current columns name of the df

def handling_missing_values(df):
    '''
        Replace missing values with 0 and log the flow for further investigation 
    '''
    df.replace([np.inf, -np.inf], 0, inplace=True)
    if df[numerical_columns].isna():
        df[numerical_columns].fillna(0,inplace = True)
        # add logging
    if df[categorical_columns].isna():
        df[categorical_columns].fillna(0,inplace = True)
        # add logging

def scaling (df):
    '''
        Scale dataframe using existing standard scaler
        to be aware that the used standard scaler here was trained on the existing dataset , ther is q risk of dqtq drift in case our data distrebution change ( possible retain to fit our own data distrbution ) 
    '''
    scaler = joblib.load('./standard_scaler.pkl')
    df_scaled = scaler.transform(df[numerical_columns])
    # change into df 
    df_scaled = pd.DataFrame(df_scaled, columns=numerical_columns)
    # add categorical to the scaled df
    df_final = pd.concat([df_scaled, df[categorical_columns].reset_index(drop=True)], axis=1)
    return df_final

def prediction_all_features(df):
    catboost_bin_uri = "runs:/e2a1ddcea0844bd6b20100df3968e891/model"
    catboost_binary = mlflow.pyfunc.load_model(catboost_bin_uri)
    catboost_mc_uri = "runs:/55b76e4143a74081868fed6e26efd638/model"
    catboost_multiclass = mlflow.pyfunc.load_model(catboost_mc_uri)

    pred_binary = catboost_binary.predict(df)
    pred_multiclass = catboost_multiclass.predict(df)
    return pred_binary, pred_multiclass

def prediction_features_selected(df):
    catboost_bin_fs_uri = "runs:/b11373e9bf37406d88c7973a52a56637/model"
    catboost_bin_fs = mlflow.pyfunc.load_model(catboost_bin_fs_uri)
    catboost_mc_uri = "runs:/3c28f4d62e3949f6aaad60e60931450d/model"
    catboost_multiclass_fs = mlflow.pyfunc.load_model(catboost_mc_uri)

    pred_binary = catboost_bin_fs.predict(df[feature_selected_binary])
    pred_multiclass = catboost_multiclass_fs.predict(df[feature_selected_multiclass])
    return pred_binary, pred_multiclass

def reverse_label_encoding(df):
    # Load the saved LabelEncoder
    le = joblib.load("label_encoder.pkl")
    df['pred_multiclass_all_features'] = le.inverse_transform(df['pred_multiclass_all_features'])
    df['pred_multiclass_fs'] = le.inverse_transform(df['pred_multiclass_fs'])
    return df

def send_results(df):
    # send elasticsearch
    print(f"timestamp : {df['timestamp']} , source ip : {df["src_ip"]} , binary : {df["pred_binary_all_features"]} , multiclass : {df["pred_multiclass_all_features"]}")
    

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def handle_post_request():
    # Get the JSON or form data from the request
    data_json = request.get_json() if request.is_json else request.form
    incoming_flow = pd.DataFrame(data_json)
    rename_columns(incoming_flow)
    handling_missing_values(incoming_flow)
    scaled_flow_df = scaling(incoming_flow)

    pred_binary_all_features , pred_multiclass_all_features = prediction_all_features(scaled_flow_df)
    pred_binary_fs , pred_multiclass_fs = prediction_features_selected(scaled_flow_df)

    incoming_flow['pred_binary_all_features'] = pred_binary_all_features
    incoming_flow['pred_multiclass_all_features'] = pred_multiclass_all_features
    incoming_flow['pred_binary_fs'] = pred_binary_fs
    incoming_flow['pred_multiclass_fs'] = pred_multiclass_fs
    incoming_flow = reverse_label_encoding(incoming_flow)

    send_results(incoming_flow)

if __name__ == '__main__':
    # Run the app in debug mode for development
    app.run(host='0.0.0.0', port=80, debug=True)