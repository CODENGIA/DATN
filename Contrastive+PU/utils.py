import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import warnings

# Tắt cảnh báo của Pandas
warnings.filterwarnings("ignore")

def prepare_dual_graph_data(pkl_path, orig_tsv_path, target_tsv_path, device='cpu'):
    """
    Hàm đọc và lọc đồ thị: 
    Lấy ma trận 11k gen (từ pkl) và lọc ra chỉ còn 6.5k gen (từ target_tsv).
    Tự động build lại danh sách các cạnh (edges) tương ứng.
    """
    print(f"--- Đang nạp và lọc dữ liệu đồ thị ---")
    
    # 1. Đọc danh sách gen gốc (11k gen)
    orig_df = pd.read_csv(orig_tsv_path, sep='\t')
    orig_genes = [str(g).strip().upper() for g in orig_df.iloc[:, 0].tolist()]
    
    # 2. Đọc danh sách gen mục tiêu cần lọc (6.5k gen)
    target_df = pd.read_csv(target_tsv_path, sep='\t')
    target_genes = [str(g).strip().upper() for g in target_df.iloc[:, 0].tolist()]
    
    # 3. Tạo từ điển Mapping index (Vị trí cũ -> Vị trí mới)
    old_to_new_idx = {}
    valid_genes = []
    new_idx = 0
    
    for old_idx, gene in enumerate(orig_genes):
        if gene in target_genes:
            old_to_new_idx[old_idx] = new_idx
            valid_genes.append(gene)
            new_idx += 1
            
    print(f"[*] Đã khớp được {len(valid_genes)} gen từ danh sách mục tiêu vào đồ thị gốc.")
    
    # 4. Đọc dữ liệu từ file PKL (11k gen)
    data = pd.read_pickle(pkl_path)
    old_norm = data['subtype_x']['Normal'].values
    old_tumor = data['subtype_x']['Tumor'].values
    old_edge_index = np.array(data['edge_index'])
    
    # 5. Gọt bớt ma trận Features (Chỉ giữ lại những hàng thuộc target_genes)
    keep_old_indices = list(old_to_new_idx.keys())
    new_norm = old_norm[keep_old_indices]
    new_tumor = old_tumor[keep_old_indices]
    
    # Chuẩn hóa Z-score trên tập đã lọc
    scaler = StandardScaler()
    new_norm_scaled = scaler.fit_transform(new_norm)
    new_tumor_scaled = scaler.transform(new_tumor)
    
    x_n = torch.tensor(new_norm_scaled, dtype=torch.float32).to(device)
    x_t = torch.tensor(new_tumor_scaled, dtype=torch.float32).to(device)
    
    # 6. Lọc và xây dựng lại ma trận Cạnh (Edge Index)
    new_edges = []
    for i in range(old_edge_index.shape[1]):
        src = old_edge_index[0, i]
        dst = old_edge_index[1, i]
        # Chỉ giữ lại cạnh nếu CẢ HAI gen đầu mút đều nằm trong danh sách 6.5k
        if src in old_to_new_idx and dst in old_to_new_idx:
            new_edges.append([old_to_new_idx[src], old_to_new_idx[dst]])
            
    new_edge_index = np.array(new_edges).T if len(new_edges) > 0 else np.empty((2, 0))
    print(f"[*] Số lượng cạnh (edges) giảm từ {old_edge_index.shape[1]} xuống {new_edge_index.shape[1]}.")
    
    e_index = torch.tensor(new_edge_index, dtype=torch.long).to(device)
    
    return x_n, e_index, x_t, e_index, valid_genes

def load_and_split_ground_truth(ncg_path, oncokb_path, current_gene_list, train_ratio=0.7, seed=42, device='cpu'):
    """
    Nạp danh sách Ground Truth (NCG và OncoKB), sau đó CHIA TẬP TRAIN/TEST trực tiếp để tránh Data Leakage.
    """
    print(f"--- Đang nạp và phân chia danh sách đáp án chuẩn (NCG & OncoKB) ---")
    driver_genes = set()
    
    try:
        ncg_df = pd.read_csv(ncg_path, sep='\t')
        ncg_genes = ncg_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        driver_genes.update(ncg_genes)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi đọc NCG: {e}")
        
    try:
        oncokb_df = pd.read_csv(oncokb_path, sep='\t')
        oncokb_genes = oncokb_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        driver_genes.update(oncokb_genes)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi đọc OncoKB: {e}")
        
    print(f"[*] Tổng số gen ung thư trong từ điển NCG và OncoKB: {len(driver_genes)}")
    
    Y_full = np.zeros(len(current_gene_list), dtype=np.float32)
    match_count = 0
    for i, gene in enumerate(current_gene_list):
        if gene in driver_genes:
            Y_full[i] = 1.0
            match_count += 1
            
    print(f"[*] Đã khớp {match_count} gen thủ phạm ung thư vào mạng lưới hiện tại.")
    
    # =========================================================
    # CHIA TẬP POSITIVE ĐỂ TRÁNH GIAN LẬN (TRANSDUCTIVE SPLIT)
    # =========================================================
    np.random.seed(seed)
    pos_indices = np.where(Y_full == 1.0)[0]
    np.random.shuffle(pos_indices) # Xáo trộn ngẫu nhiên
    
    num_train = int(len(pos_indices) * train_ratio)
    train_pos_idx = pos_indices[:num_train]
    test_pos_idx = pos_indices[num_train:]
    
    # 1. Y_train: Chỉ cho mô hình biết một phần gen ung thư. Phần còn lại coi là 0 (Unlabeled)
    Y_train = np.zeros_like(Y_full)
    Y_train[train_pos_idx] = 1.0
    
    # 2. eval_mask: Mặt nạ chỉ định những gen nào được phép dùng để chấm điểm.
    eval_mask = np.ones_like(Y_full, dtype=bool)
    eval_mask[train_pos_idx] = False
    
    print(f"[*] Đã cấp: {len(train_pos_idx)} gen Positive cho Train (mớm cho nnPU)")
    print(f"[*] Đã giấu: {len(test_pos_idx)} gen Positive cho Test (để chấm AUPRC khách quan)")
    
    return (torch.tensor(Y_train, dtype=torch.float32).to(device), 
            torch.tensor(Y_full, dtype=torch.float32).to(device), 
            torch.tensor(eval_mask, dtype=torch.bool).to(device))