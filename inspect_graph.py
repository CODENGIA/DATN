import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_dual_graph_data(pkl_path, tsv_path, true_path, false_path, device='cpu'):
   # 1. Nạp dữ liệu
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # Đọc file TSV bình thường, không gán index_col
    gene_index_df = pd.read_csv(tsv_path, sep='\t')
    
    # [FIX CỰC MẠNH Ở ĐÂY]: 
    # .iloc[:, 0] nghĩa là chỉ bốc đúng Cột số 0 (cột chứa tên gen)
    # Bỏ qua hoàn toàn Cột số 1 (cột chứa số 1409, 1410...)
    raw_gene_list = gene_index_df.iloc[:, 0].tolist()
    
    # Ép kiểu chuỗi, xóa mọi khoảng trắng thừa và viết IN HOA
    current_gene_list = [str(g).strip().upper() for g in raw_gene_list]
    
    
    # 2. Xử lý Node Features
    features_dict = data['subtype_x']
    scaler = StandardScaler()
    scaler.fit(features_dict['Normal'].values) 
    
    x_norm = torch.tensor(scaler.transform(features_dict['Normal'].values), dtype=torch.float32).to(device)
    x_tumor = torch.tensor(scaler.transform(features_dict['Tumor'].values), dtype=torch.float32).to(device)
    
    # 3. Xử lý Edge Index
    edge_index = torch.tensor(np.array(data['edge_index']), dtype=torch.long).to(device)
    
    # 4. KHỞI TẠO NHÃN Y
    # Đọc file TXT, ép IN HOA và xóa khoảng trắng để đảm bảo khớp 100%
    with open(true_path, 'r') as f:
        true_genes = set(line.strip().upper() for line in f)
    with open(false_path, 'r') as f:
        false_genes = set(line.strip().upper() for line in f)
    
    # Tạo vector Y ban đầu là -1
    N = len(current_gene_list)
    Y = np.full(N, -1, dtype=np.float32) 
    
    match_true = 0
    match_false = 0
    
    # Đối chiếu
    for i, gene in enumerate(current_gene_list):
        if gene in true_genes:
            Y[i] = 1.0
            match_true += 1
        elif gene in false_genes:
            Y[i] = 0.0
            match_false += 1
            
    print(f"\n--- BÁO CÁO GÁN NHÂN ---")
    print(f"Tổng số gen trong mạng lưới: {N}")
    print(f"Khớp được {match_true} gen Driver (Nhãn 1)")
    print(f"Khớp được {match_false} gen Passenger (Nhãn 0)")
    print(f"Số gen chưa có nhãn (Sẽ gán -1): {N - match_true - match_false}")
            
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    labeled_mask = (Y != -1)
    
    return x_norm, edge_index, x_tumor, edge_index, Y, labeled_mask, gene_index_df

# --- GỌI THỬ HÀM ---
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = r"E:\DATN\GRAFT\Data" # Bạn thay bằng đường dẫn thực tế của mình nhé
    
    x_n, e_n, x_t, e_t, Y, mask, g_df = prepare_dual_graph_data(
        pkl_path = f"{base_path}\\LUAD_input_data_humannet.pkl",
        tsv_path = f"{base_path}\\LUAD_gene_index_humannet.tsv",
        true_path = f"{base_path}\\796true.txt",
        false_path = f"{base_path}\\2187false.txt",
        device = device
    )
except Exception as e:
    print(f"Lỗi: {e}")