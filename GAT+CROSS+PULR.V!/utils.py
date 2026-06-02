import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def prepare_dual_graph_data(pkl_path, orig_tsv_path, target_tsv_path, device='cpu'):
    """
    Load và lọc dữ liệu đồ thị kép (Normal + Tumor).

    Pipeline:
        1. Đọc danh sách gen gốc (11k) và gen mục tiêu (6.5k)
        2. Build mapping old_idx -> new_idx
        3. Cắt ma trận features, chuẩn hóa Z-score (scaler fit từ x_n)
        4. Lọc edges (giữ cạnh khi CẢ HAI đầu mút thuộc target_genes)
        5. Trả về e_n và e_t là 2 biến RIÊNG BIỆT (sẵn sàng cho multiplex graph)

    Returns:
        x_n     : Tensor [N, 14] - features Normal (đã chuẩn hóa)
        e_n     : Tensor [2, E_n] - edge index Normal
        x_t     : Tensor [N, 14] - features Tumor (chuẩn hóa bằng scaler của Normal)
        e_t     : Tensor [2, E_t] - edge index Tumor (hiện tại = e_n, tách biến riêng)
        valid_genes : list[str]  - danh sách gen theo thứ tự hàng của x_n / x_t
    """
    print("--- Đang nạp và lọc dữ liệu đồ thị ---")

    # --- 1. Đọc danh sách gen ---
    orig_df    = pd.read_csv(orig_tsv_path, sep='\t')
    orig_genes = [str(g).strip().upper() for g in orig_df.iloc[:, 0].tolist()]

    target_df    = pd.read_csv(target_tsv_path, sep='\t')
    target_genes = set(str(g).strip().upper() for g in target_df.iloc[:, 0].tolist())

    # --- 2. Build mapping old_idx -> new_idx ---
    old_to_new_idx = {}
    valid_genes    = []
    new_idx        = 0

    for old_idx, gene in enumerate(orig_genes):
        if gene in target_genes:
            old_to_new_idx[old_idx] = new_idx
            valid_genes.append(gene)
            new_idx += 1

    print(f"[*] Đã khớp được {len(valid_genes)} gen từ danh sách mục tiêu vào đồ thị gốc.")

    # --- 3. Đọc PKL, cắt features, chuẩn hóa Z-score ---
    data      = pd.read_pickle(pkl_path)
    old_norm  = data['subtype_x']['Normal'].values   # [11k, 14]
    old_tumor = data['subtype_x']['Tumor'].values    # [11k, 14]
    old_edge_index = np.array(data['edge_index'])    # [2, E_orig]

    keep_old_indices = list(old_to_new_idx.keys())
    new_norm  = old_norm[keep_old_indices]            # [N, 14]
    new_tumor = old_tumor[keep_old_indices]           # [N, 14]

    # Z-score: fit từ Normal, transform cả 2 (đúng thứ tự sinh học)
    scaler         = StandardScaler()
    new_norm_sc    = scaler.fit_transform(new_norm)
    new_tumor_sc   = scaler.transform(new_tumor)

    x_n = torch.tensor(new_norm_sc,  dtype=torch.float32).to(device)
    x_t = torch.tensor(new_tumor_sc, dtype=torch.float32).to(device)

    # --- 4. Lọc và xây dựng lại edges ---
    new_edges = []
    for i in range(old_edge_index.shape[1]):
        src, dst = old_edge_index[0, i], old_edge_index[1, i]
        if src in old_to_new_idx and dst in old_to_new_idx:
            new_edges.append([old_to_new_idx[src], old_to_new_idx[dst]])

    if len(new_edges) > 0:
        new_edge_index = np.array(new_edges, dtype=np.int64).T  # [2, E_new]
    else:
        new_edge_index = np.empty((2, 0), dtype=np.int64)

    print(f"[*] Edges: {old_edge_index.shape[1]} -> {new_edge_index.shape[1]} sau khi lọc.")

    e_index = torch.tensor(new_edge_index, dtype=torch.long).to(device)

    # --- 5. Tách biến e_n và e_t để sẵn sàng multiplex graph sau này ---
    e_n = e_index          # Normal edge index
    e_t = e_index.clone()  # Tumor edge index (clone để tách biến độc lập trong memory)

    return x_n, e_n, x_t, e_t, valid_genes


def load_and_split_ground_truth(ncg_path, oncokb_path, current_gene_list,
                                 train_ratio=0.7, val_ratio=0.15, seed=42, device='cpu'):
    """
    Load danh sách Ground Truth (NCG + OncoKB) và chia Transductive Split.

    Phân chia:
        train_ratio : tỷ lệ Positive đưa vào Train (mặc định 70%)
        val_ratio   : tỷ lệ Positive giữ cho Val   (mặc định 15%)
        phần còn lại: Test (15%)

    Returns:
        Y_train    : Tensor [N] - chỉ chứa train_pos, còn lại = 0
        Y_val      : Tensor [N] - chỉ chứa val_pos, còn lại = 0
        Y_test     : Tensor [N] - chỉ chứa test_pos, còn lại = 0
        Y_full     : Tensor [N] - toàn bộ nhãn thật (dùng để đánh giá)
        train_mask : BoolTensor [N] - True tại các gen train pos
        val_mask   : BoolTensor [N] - True tại các gen val pos + unlabeled
        test_mask  : BoolTensor [N] - True tại các gen test pos + unlabeled
    """
    print("--- Đang nạp và phân chia danh sách Ground Truth (NCG & OncoKB) ---")

    driver_genes = set()

    # Load NCG
    try:
        ncg_df = pd.read_csv(ncg_path, sep='\t')
        ncg_genes = ncg_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        driver_genes.update(ncg_genes)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi đọc NCG: {e}")

    # Load OncoKB
    try:
        oncokb_df = pd.read_csv(oncokb_path, sep='\t')
        oncokb_genes = oncokb_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        driver_genes.update(oncokb_genes)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi đọc OncoKB: {e}")

    print(f"[*] Tổng gen ung thư trong NCG + OncoKB: {len(driver_genes)}")

    # Build Y_full
    Y_full = np.zeros(len(current_gene_list), dtype=np.float32)
    for i, gene in enumerate(current_gene_list):
        if gene in driver_genes:
            Y_full[i] = 1.0

    match_count = int(Y_full.sum())
    print(f"[*] Đã khớp {match_count} gen ung thư vào mạng lưới ({len(current_gene_list)} gen).")

    # ============================
    # TRANSDUCTIVE SPLIT (70/15/15)
    # ============================
    np.random.seed(seed)
    pos_indices = np.where(Y_full == 1.0)[0]
    np.random.shuffle(pos_indices)

    n_total = len(pos_indices)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    # n_test  = n_total - n_train - n_val  (phần còn lại)

    train_pos_idx = pos_indices[:n_train]
    val_pos_idx   = pos_indices[n_train : n_train + n_val]
    test_pos_idx  = pos_indices[n_train + n_val :]

    print(f"[*] Train Positive : {len(train_pos_idx)} gen (cho nnPU học)")
    print(f"[*] Val   Positive : {len(val_pos_idx)}   gen (Early Stopping theo AUPRC)")
    print(f"[*] Test  Positive : {len(test_pos_idx)}  gen (Đánh giá cuối cùng)")

    # Y_train: chỉ chứa train_pos, phần còn lại = 0 (Unlabeled)
    Y_train = np.zeros_like(Y_full)
    Y_train[train_pos_idx] = 1.0

    # Y_val, Y_test: dùng để chấm điểm
    Y_val  = Y_full.copy()
    Y_test = Y_full.copy()

    # Mask để lọc đúng tập khi evaluate
    # val_mask  : loại bỏ train_pos (không được chấm)
    # test_mask : loại bỏ train_pos và val_pos (chỉ chấm gen thật sự ẩn)
    N = len(current_gene_list)
    val_mask  = np.ones(N, dtype=bool)
    val_mask[train_pos_idx] = False

    test_mask = np.ones(N, dtype=bool)
    test_mask[train_pos_idx] = False
    test_mask[val_pos_idx]   = False

    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).to(device)

    return (
        to_tensor(Y_train),
        to_tensor(Y_val),
        to_tensor(Y_test),
        to_tensor(Y_full),
        to_tensor(val_mask,  dtype=torch.bool),
        to_tensor(test_mask, dtype=torch.bool),
    )
