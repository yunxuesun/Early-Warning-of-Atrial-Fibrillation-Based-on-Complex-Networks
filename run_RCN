# run_RCN.py
import os
import numpy as np
import pandas as pd
import wfdb
import networkx as nx
import hrvanalysis as hrv
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation, RPComputation

warnings.filterwarnings("ignore")

DATA_PATH = './paf-prediction-challenge-database-1.0.0/'
PRO_DATA_DIR = './pro_data/'
SAMPLING_FREQUENCY = 128

def clean_rr(rr_list, remove_invalid=True, low_rr=0.2, high_rr=4, interpolation_method="linear", remove_ecto=True) -> np.ndarray:
    if remove_invalid:
        rr_list = [rr if high_rr >= rr >= low_rr else np.nan for rr in rr_list]
        rr_list = pd.Series(rr_list).interpolate(method=interpolation_method).tolist()
    if remove_ecto:
        rr_list = hrv.remove_ectopic_beats(rr_list, method='custom', custom_removing_rule=0.3, verbose=False)
        rr_list = pd.Series(rr_list).interpolate(method=interpolation_method).interpolate(limit_direction='both').tolist()
    return np.array(rr_list)

def extract_rcn_data():
    """提取复杂网络需要清洗的RR间期"""
    os.makedirs(PRO_DATA_DIR, exist_ok=True)
    nsr_path = os.path.join(PRO_DATA_DIR, 'NSR_RR_300.npy')
    preaf_path = os.path.join(PRO_DATA_DIR, 'PreAF_RR_300.npy')
    
    if os.path.exists(nsr_path) and os.path.exists(preaf_path):
        print("[RCN] 预处理数据(300长度)已存在，跳过提取...")
        return
        
    print("[RCN] 正在进行带降噪清洗的 RR 间期提取...")
    nsr_rri, preaf_rri = [], []
    
    for i in tqdm(range(1, 51, 2), desc="提取 NSR"):
        record_name = os.path.join(DATA_PATH, f'p{i:02d}')
        if os.path.exists(record_name + '.qrs'):
            annotation = wfdb.rdann(record_name, 'qrs')
            rr_intervals = clean_rr(np.diff(annotation.sample) / SAMPLING_FREQUENCY)
            for j in range(3):
                nsr_rri.append(rr_intervals[31*j:31*(j+1)]) 
                
    for i in tqdm(range(2, 51, 2), desc="提取 PreAF"):
        record_name = os.path.join(DATA_PATH, f'p{i:02d}')
        if os.path.exists(record_name + '.qrs'):
            annotation = wfdb.rdann(record_name, 'qrs')
            rr_intervals = clean_rr(np.diff(annotation.sample) / SAMPLING_FREQUENCY)
            for j in range(3):
                start_index = len(rr_intervals) - 300 * (j + 1)
                end_index = len(rr_intervals) - 300 * j
                if start_index >= 0:
                    preaf_rri.append(rr_intervals[start_index:end_index])
                    
    np.save(nsr_path, nsr_rri)
    np.save(preaf_path, preaf_rri)
    print("[RCN] 数据清洗与保存完成！")

def compute_kth_moment(adj_matrix, k):
    eigenvalues = np.linalg.eigvals(adj_matrix)
    eigenvalues_k = np.power(eigenvalues, k)
    return (np.sum(eigenvalues_k) / adj_matrix.shape[0]).real

def get_rqa_all(file_path, threshold=0.24):
    rqa = []
    data = np.load(file_path, allow_pickle=True)
    filename = os.path.basename(file_path)
    
    for i in tqdm(range(len(data)), desc=f"提取特征: {filename}"):
        time_series = TimeSeries(data[i], embedding_dimension=3, time_delay=2)
        settings = Settings(time_series, analysis_type=Classic, 
                            neighbourhood=FixedRadius(threshold), 
                            similarity_measure=EuclideanMetric, theiler_corrector=1)
        
        # 复杂网络拓扑特征
        result_rp = RPComputation.create(settings, verbose=False).run()
        matrix = result_rp.recurrence_matrix
        G = nx.from_numpy_array(matrix)
        average_degree = sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes)>0 else 0
        avg_clustering = nx.average_clustering(G)
        moment_3th = compute_kth_moment(matrix, 3)

        # 基础RQA特征
        result_rqa = RQAComputation.create(settings, verbose=False).run()
        rqa.append([
            average_degree, avg_clustering, moment_3th, 
            result_rqa.recurrence_rate, result_rqa.entropy_diagonal_lines, 
            result_rqa.determinism, result_rqa.laminarity
        ])
    return np.array(rqa)

def main():
    print("\n" + "="*50)
    print("模块二：递归复杂网络 (RCN) 与特征融合模型启动")
    print("="*50)
    extract_rcn_data()
    
    print("\n[RCN] 开始提取 7 维融合特征 (RQA + RCN)...")
    X_nsr = get_rqa_all(os.path.join(PRO_DATA_DIR, 'NSR_RR_300.npy'), threshold=0.24)
    X_preaf = get_rqa_all(os.path.join(PRO_DATA_DIR, 'PreAF_RR_300.npy'), threshold=0.24)
    
    X = np.vstack((X_nsr, X_preaf))
    y = np.hstack((np.zeros(len(X_nsr)), np.ones(len(X_preaf))))
    
    print("\n[RCN] 开始训练 XGBoost 特征融合分类器...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = xgb.XGBClassifier(eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"\n[评估结果] 特征融合模型准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("[评估结果] 详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['SR (0)', 'Pre-AF (1)']))

if __name__ == "__main__":
    main()
