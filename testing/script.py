from flask import Flask, request, jsonify
import mlflow
import numpy as np
import pandas as pd
import joblib
from elasticsearch import Elasticsearch # type: ignore
from elasticsearch.helpers import bulk # type: ignore

mlflow.set_tracking_uri("http://localhost:5000")



#step1 mapping the columns names of CICflowmeter json output to match preused columns names in the training dataset 

rename_dict = {'dst_port': 'Destination Port','flow_duration': 'Flow Duration','tot_fwd_pkts': 'Total Fwd Packets','tot_bwd_pkts': 'Total Backward Packets','totlen_fwd_pkts': 'Total Length of Fwd Packets','totlen_bwd_pkts': 'Total Length of Bwd Packets','fwd_pkt_len_max': 'Fwd Packet Length Max','fwd_pkt_len_min': 'Fwd Packet Length Min','fwd_pkt_len_mean': 'Fwd Packet Length Mean','fwd_pkt_len_std': 'Fwd Packet Length Std','bwd_pkt_len_max': 'Bwd Packet Length Max','bwd_pkt_len_min': 'Bwd Packet Length Min','bwd_pkt_len_mean': 'Bwd Packet Length Mean','bwd_pkt_len_std': 'Bwd Packet Length Std','flow_byts_s': 'Flow Bytes/s','flow_pkts_s': 'Flow Packets/s','fwd_iat_mean': 'Flow IAT Mean','fwd_iat_std': 'Flow IAT Std','fwd_iat_max': 'Flow IAT Max','fwd_iat_min': 'Flow IAT Min','fwd_iat_tot': 'Fwd IAT Total','flow_iat_mean': 'Fwd IAT Mean','flow_iat_std': 'Fwd IAT Std','flow_iat_max': 'Fwd IAT Max','flow_iat_min': 'Fwd IAT Min','bwd_iat_tot': 'Bwd IAT Total','bwd_iat_mean': 'Bwd IAT Mean','bwd_iat_std': 'Bwd IAT Std','bwd_iat_max': 'Bwd IAT Max','bwd_iat_min': 'Bwd IAT Min','fwd_psh_flags': 'Fwd PSH Flags','fwd_urg_flags': 'Fwd URG Flags','fwd_header_len': 'Fwd Header Length','bwd_header_len': 'Bwd Header Length','fwd_pkts_s': 'Fwd Packets/s','bwd_pkts_s': 'Bwd Packets/s','pkt_len_min': 'Min Packet Length','pkt_len_max': 'Max Packet Length','pkt_len_mean': 'Packet Length Mean','pkt_len_std': 'Packet Length Std','pkt_len_var': 'Packet Length Variance','fin_flag_cnt': 'FIN Flag Count','rst_flag_cnt': 'RST Flag Count','psh_flag_cnt': 'PSH Flag Count','ack_flag_cnt': 'ACK Flag Count','urg_flag_cnt': 'URG Flag Count','ece_flag_cnt': 'ECE Flag Count','down_up_ratio': 'Down/Up Ratio','pkt_size_avg': 'Average Packet Size','fwd_seg_size_avg': 'Avg Fwd Segment Size','bwd_seg_size_avg': 'Avg Bwd Segment Size','subflow_fwd_byts': 'Subflow Fwd Bytes','subflow_bwd_byts': 'Subflow Bwd Bytes','init_fwd_win_byts': 'Init_Win_bytes_forward','init_bwd_win_byts': 'Init_Win_bytes_backward','fwd_act_data_pkts': 'act_data_pkt_fwd','fwd_seg_size_min': 'min_seg_size_forward','active_mean': 'Active Mean','active_std': 'Active Std','active_max': 'Active Max','active_min': 'Active Min','idle_mean': 'Idle Mean','idle_std': 'Idle Std','idle_max': 'Idle Max','idle_min': 'Idle Min'}

def rename_columns(incoming_df):#renaming yhe columns of the incoming dataframe
    incoming_df.rename(columns=rename_dict, inplace=True) # we change the current columns name of the df




# step2: handling missing values in the incoming flows from CICflowmeter


numerical_columns = ["Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Subflow Fwd Bytes","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"]
categorical_columns= ['Fwd PSH Flags','Fwd URG Flags','FIN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','ECE Flag Count']


def handling_missing_values(df):
    '''
        Replace missing values with 0 and log the flow for further investigation 
    '''
    df.replace([np.inf, -np.inf], 0, inplace=True)
    if df[numerical_columns].isna().any().any():
        df[numerical_columns].fillna(0,inplace = True)
        # add logging
    if df[categorical_columns].isna().any().any():
        df[categorical_columns].fillna(0,inplace = True)
        # add logging


# step3: Scale the output data from the cicflowmeter using the pretrained scaling model used in the feature engineering steps


def scaling (df):
    '''
        Scale dataframe using existing standard scaler
        to be aware that the used standard scaler here was trained on the existing dataset , ther is a risk of data drift in case our data distribution change ( possible retain to fit our own data distrbution ) 
    '''
    scaler = joblib.load('./standard_scaler.pkl')
    df_scaled = scaler.transform(df[numerical_columns])
    # change into df 
    df_scaled = pd.DataFrame(df_scaled, columns=numerical_columns)
    # add categorical to the scaled df
    df_final = pd.concat([df_scaled, df[categorical_columns].reset_index(drop=True)], axis=1)
    return df_final

# step4: load the selected models from MLflow for inline prediction


feature_selected_binary = ['Average Packet Size', 'Packet Length Variance', 'Packet Length Std', 'Destination Port', 'Packet Length Mean', 'Total Length of Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Max Packet Length', 'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Init_Win_bytes_backward', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Flow IAT Max', 'Flow Bytes/s', 'Flow Duration', 'Bwd Packets/s', 'Bwd Header Length', 'Fwd Packets/s', 'Fwd IAT Max', 'Flow Packets/s', 'Fwd Header Length', 'Flow IAT Mean', 'Bwd IAT Max', 'Fwd IAT Total', 'Bwd Packet Length Std', 'Bwd IAT Total', 'Fwd IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean', 'Flow IAT Std', 'Total Backward Packets', 'Fwd Packet Length Min', 'Min Packet Length', 'Bwd Packet Length Min', 'Fwd IAT Std', 'Active Min', 'Active Mean', 'Active Max']
feature_selected_multiclass = ['Packet Length Variance', 'Packet Length Std', 'Packet Length Mean', 'Average Packet Size', 'Total Length of Bwd Packets', 'Subflow Bwd Bytes', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Init_Win_bytes_backward', 'Destination Port', 'Max Packet Length', 'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Bwd Packet Length Std', 'Flow IAT Max', 'Init_Win_bytes_forward', 'Fwd IAT Max', 'Bwd Packets/s', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Flow Bytes/s', 'Fwd Packet Length Std', 'Fwd Header Length', 'Flow Packets/s', 'Flow IAT Mean', 'Flow Duration', 'Bwd Header Length', 'Fwd Packets/s', 'Fwd IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Std', 'Bwd IAT Max', 'Total Backward Packets', 'Bwd IAT Total', 'Bwd IAT Mean', 'Total Fwd Packets', 'Bwd IAT Std', 'Active Min', 'Active Mean', 'Idle Max', 'Active Max', 'Bwd IAT Min']


catboost_bin_uri = 'runs:/e2a1ddcea0844bd6b20100df3968e891/CatBoost_model_mlflow'
catboost_binary = mlflow.pyfunc.load_model(catboost_bin_uri)
catboost_mc_uri =  'runs:/55b76e4143a74081868fed6e26efd638/CatBoost_model_mlflow'
catboost_multiclass = mlflow.pyfunc.load_model(catboost_mc_uri)
catboost_bin_fs_uri = 'runs:/b11373e9bf37406d88c7973a52a56637/CatBoost_model_mlflow'
catboost_bin_fs = mlflow.pyfunc.load_model(catboost_bin_fs_uri)
catboost_mc_uri = 'runs:/5b69acb463d640eeb15081d7bfbc25c5/CatBoost_model_mlflow'
catboost_multiclass_fs = mlflow.pyfunc.load_model(catboost_mc_uri)

def prediction_all_features(df):
    pred_binary = catboost_binary.predict(df)
    pred_multiclass = catboost_multiclass.predict(df)
    return pred_binary, pred_multiclass

def prediction_features_selected(df):
    pred_binary = catboost_bin_fs.predict(df[feature_selected_binary])
    pred_multiclass = catboost_multiclass_fs.predict(df[feature_selected_multiclass])
    return pred_binary, pred_multiclass


# step5: preparing  and formatting (exp, the timestamp format "datetime" + reverse encoding of labels) the data (output flow from CICflowmeter + models predictions) to be sent to the SIEM

def reverse_label_encoding(df):
    # Load the saved LabelEncoder
    le = joblib.load("label_encoder.pkl")
    df['pred_multiclass_all_features'] = le.inverse_transform(df['pred_multiclass_all_features'])
    df['pred_multiclass_fs'] = le.inverse_transform(df['pred_multiclass_fs'])
    return df


# step6: send the result to the SIEM to be able to visualize the output in elasticsearch platform
def send_results(df, doc_type='_doc'):
    print(f"timestamp : {df['timestamp'].iloc[0]} , source ip : {df['src_ip'].iloc[0]} , multiclass-all : {df['pred_multiclass_all_features'].iloc[0]} , multiclass-fs : {df['pred_multiclass_fs'].iloc[0]} , binary-fs : {df['pred_binary_fs'].iloc[0]}, binary-all : {df['pred_binary_all_features'].iloc[0]}")
    try:
        # Connect to Elasticsearch
        index_name = 'cic'
        es_ip = '172.16.27.225'
        es_port = 9200
        # Add username and password
        es_username = 'asmae'  # Replace with your actual username
        es_password = 'hbJJqncbMGEjuZWPOd6K'  # Replace with your actual password
        es = Elasticsearch(
            [f'http://{es_ip}:{es_port}'],
            basic_auth=(es_username, es_password),
            verify_certs=False
        )
        # Rest of your code remains the same
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, ignore=400)
        actions = [
            {
                "_index": index_name,
                "_op_type": "index",
                "_source": row.to_dict(),
            }
            for _, row in df.iterrows()
        ]
        success, failed = bulk(es, actions)
        print(f"Indexed {success} documents successfully. Failed to index {failed} documents.")
 
    except Exception as e:
        print(f"**********************An error occurred: {str(e)}")     

# ===========================end of step6================

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def handle_post_request():

    # step 0 : receiving the flow from CICflowmeter and converting it to a dataframe
    data_json = request.get_json() if request.is_json else request.form
    incoming_flow = pd.DataFrame([data_json])


    #step1: Map Column Names
    rename_columns(incoming_flow)

    #step2: Handle Missing Values
    handling_missing_values(incoming_flow)

    # step3: Scale Data
    scaled_flow_df = scaling(incoming_flow)

    # step4: 4.	Load Models for Prediction
    pred_binary_all_features , pred_multiclass_all_features = prediction_all_features(scaled_flow_df)
    pred_binary_fs , pred_multiclass_fs = prediction_features_selected(scaled_flow_df)

    # step5: Data Preparation and Formatting
    incoming_flow['pred_binary_all_features'] = pred_binary_all_features
    incoming_flow['pred_multiclass_all_features'] = pred_multiclass_all_features
    incoming_flow['pred_binary_fs'] = pred_binary_fs
    incoming_flow['pred_multiclass_fs'] = pred_multiclass_fs
    incoming_flow = reverse_label_encoding(incoming_flow)
    incoming_flow['timestamp'] = pd.to_datetime(incoming_flow['timestamp']).dt.tz_localize('UTC').dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    #step6: Send Results to SIEM
    send_results(incoming_flow) #test

    response = {
        "status": "success",
        "message": "Data received successfully!"
    }

    return jsonify(response), 200  # Return JSON with a 200 OK status

if __name__ == '__main__':
    # Run the app in debug mode for development
    app.run(host='0.0.0.0', port=5008, debug=True)
