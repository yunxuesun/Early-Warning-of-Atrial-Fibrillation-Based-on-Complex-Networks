# run_RQA.py
import os
import numpy as np
import wfdb
import xgboost as xgb
import warnings
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 忽略环境警告，保持终端输出干净
warnings.filterwarnings("ignore")

DATA_PATH = './paf-prediction-challenge-database-1.0.0/'
PRO_DATA_DIR = './pro_data/'
SAMPLING_FREQUENCY = 128

def extract_rqa_data():
    """提取RR间期并保存"""
    os.makedirs(PRO_DATA_DIR, exist_ok=True)
    nsr_path = os.path.join(PRO_DATA_DIR, 'NSR_RR.npy')
    preaf_path = os.path.join(PRO_DATA_DIR, 'PreAF_RR.npy')
    
    if os.path.exists(nsr_path) and os.path.exists(preaf_path):
        print("[RQA] 预处理数据已存在，跳过提取步骤...")
        return
        
    print("[RQA] 正在从原始数据库提取 RR 间期...")
    nsr_rri, preaf_rri = [], []
    
    # 提取 NSR
    for i in range(1, 51, 2):
        record_name = os.path.join(DATA_PATH, f'p{i:02d}')
        if os.path.exists(record_name + '.qrs'):
            annotation = wfdb.rdann(record_name, 'qrs')
            rr_intervals = np.diff(annotation.sample) / SAMPLING_FREQUENCY
            for j in range(3):
                nsr_rri.append(rr_intervals[31*j:31*(j+1)])
                
    # 提取 PreAF
    for i in range(2, 51, 2):
        record_name = os.path.join(DATA_PATH, f'p{i:02d}')
        if os.path.exists(record_name + '.qrs'):
            annotation = wfdb.rdann(record_name, 'qrs')
            rr_intervals = np.diff(annotation.sample) / SAMPLING_FREQUENCY
            for j in range(3):
                start_index = len(rr_intervals) - 31 * (j + 1)
                end_index = len(rr_intervals) - 31 * j
                if start_index >= 0:
                    preaf_rri.append(rr_intervals[start_index:end_index])
                    
    np.save(nsr_path, nsr_rri)
    np.save(preaf_path, preaf_rri)
    print("[RQA] 数据提取与保存完成！")

def get_rqa_mea(file_path, threshold=0.05, measure='determinism'):
    """计算单维度RQA特征"""
    rqa = []
    data = np.load(file_path, allow_pickle=True)
    for i in range(len(data)):
        time_series = TimeSeries(data[i], embedding_dimension=3, time_delay=2)
        settings = Settings(time_series, analysis_type=Classic, 
                            neighbourhood=FixedRadius(threshold), 
                            similarity_measure=EuclideanMetric, theiler_corrector=1)
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        if measure == 'determinism':
            rqa.append(result.determinism)
    return rqa

def main():
    print("\n" + "="*50)
    print("模块一：定量递归分析 (RQA) 预测模型启动")
    print("="*50)
    extract_rqa_data()
    
    thresholds = np.arange(0.01, 0.21, 0.01)
    best_accuracy, best_threshold = 0, 0
    
    print("\n[RQA] 正在进行距离阈值寻优 (基于 determinism)，请稍候...")
    for th in tqdm(thresholds, desc="阈值寻优"):
        nsr_rqa = get_rqa_mea(os.path.join(PRO_DATA_DIR, 'NSR_RR.npy'), threshold=th)
        preaf_rqa = get_rqa_mea(os.path.join(PRO_DATA_DIR, 'PreAF_RR.npy'), threshold=th)
        
        X = np.vstack((np.array(nsr_rqa).reshape(-1, 1), np.array(preaf_rqa).reshape(-1, 1)))
        y = np.hstack((np.zeros(len(nsr_rqa)), np.ones(len(preaf_rqa))))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = xgb.XGBClassifier(eval_metric='logloss')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = th
            
    print(f"\n[RQA] 寻优完毕！最佳阈值为 {best_threshold:.2f}，最高准确率: {best_accuracy:.4f}")
    
    print("\n[RQA] 打印最佳参数下的分类报告：")
    nsr_rqa_best = get_rqa_mea(os.path.join(PRO_DATA_DIR, 'NSR_RR.npy'), threshold=best_threshold)
    preaf_rqa_best = get_rqa_mea(os.path.join(PRO_DATA_DIR, 'PreAF_RR.npy'), threshold=best_threshold)
    
    X_best = np.vstack((np.array(nsr_rqa_best).reshape(-1, 1), np.array(preaf_rqa_best).reshape(-1, 1)))
    y_best = np.hstack((np.zeros(len(nsr_rqa_best)), np.ones(len(preaf_rqa_best))))
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_best, y_best, test_size=0.2, random_state=42)
    
    clf_best = xgb.XGBClassifier(eval_metric='logloss')
    clf_best.fit(X_train_b, y_train_b)
    print(classification_report(y_test_b, clf_best.predict(X_test_b), target_names=['SR (0)', 'Pre-AF (1)']))

if __name__ == "__main__":
    main()
