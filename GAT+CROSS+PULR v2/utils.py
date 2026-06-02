"""
utils.py  —  Data Pipeline cho SiamesePUGNN v2
================================================
Nâng cấp so với phiên bản trước:
  [U1] estimate_prior_alphamax()  : Ước lượng prior động bằng AlphaMax-inspired method
  [U2] Edge-weight generation     : Tạo edge_weight tensor (Uniform(0.5, 1.0) dummy nếu
                                    file .pkl không chứa trọng số cạnh thực)
  [U3] Transductive 70/15/15 split: Trả về đủ 6 object cho train/val/test
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# [U1]  ALPHAMAX-INSPIRED PRIOR ESTIMATION
# ==============================================================================
def estimate_prior_alphamax(x_n: torch.Tensor,
                             x_t: torch.Tensor,
                             Y_train: torch.Tensor,
                             clip_low: float = 0.01,
                             clip_high: float = 0.15) -> float:
    """
    Ước lượng prior π = P(Y=1) theo tinh thần AlphaMax:
        π ≈ mean_prob(Positive) / max(mean_prob(Unlabeled), ε)
    nhưng dùng Logistic Regression nhanh thay vì kernel density.

    Quy trình:
        1. Ghép x_n và x_t thành vector 2*in_dim, chuẩn hóa.
        2. Fit LR nhị phân trên Y_train (1=Positive, 0=Unlabeled).
        3. Tính mean predicted probability cho từng nhóm.
        4. prior = mean_prob_pos / mean_prob_unl (capped trong [clip_low, clip_high]).

    Args:
        x_n       : Tensor [N, in_dim] — features Normal (trên device)
        x_t       : Tensor [N, in_dim] — features Tumor   (trên device)
        Y_train   : Tensor [N]         — nhãn train (1=known Pos, 0=Unlabeled)
        clip_low  : float              — giới hạn dưới prior (0.01)
        clip_high : float              — giới hạn trên prior (0.15)

    Returns:
        estimated_prior : float ∈ [clip_low, clip_high]
    """
    # Chuyển về CPU numpy để dùng sklearn
    X = torch.cat([x_n, x_t], dim=-1).detach().cpu().numpy()   # [N, 2*in_dim]
    y = Y_train.detach().cpu().numpy()                          # [N]

    # Chuẩn hóa lại cho LR ổn định
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pos_mask = (y == 1.0)
    unl_mask = (y == 0.0)

    num_pos = pos_mask.sum()
    num_unl = unl_mask.sum()

    if num_pos < 2 or num_unl < 2:
        print("[AlphaMax] Không đủ mẫu để ước lượng prior, dùng fallback=0.05")
        return 0.05

    # LR nhanh (max_iter=200 đủ cho feature 28-dim)
    lr_model = LogisticRegression(
        max_iter=200, solver='lbfgs', C=1.0,
        class_weight='balanced', random_state=42
    )
    lr_model.fit(X_sc, y)
    probs = lr_model.predict_proba(X_sc)[:, 1]   # P(Y=1 | x)

    mean_pos = probs[pos_mask].mean()             # Xác suất trung bình nhóm Positive
    mean_unl = probs[unl_mask].mean()             # Xác suất trung bình nhóm Unlabeled

    if mean_unl < 1e-6:
        print("[AlphaMax] mean_unl quá nhỏ, dùng fallback=0.05")
        return 0.05

    # Công thức AlphaMax đơn giản hoá
    raw_prior = float(mean_pos / (mean_unl + 1e-8))
    # Nếu tập Positive được chia đúng, ratio này ≈ 1/π
    # → prior ≈ 1/raw_prior (lấy nghịch đảo rồi clip)
    estimated_prior = float(np.clip(1.0 / (raw_prior + 1e-8), clip_low, clip_high))

    print(f"[AlphaMax] mean_prob_pos={mean_pos:.4f} | mean_prob_unl={mean_unl:.4f} "
          f"| raw_ratio={raw_prior:.4f} | estimated_prior={estimated_prior:.4f}")
    return estimated_prior


# ==============================================================================
# [U2 + U3]  DUAL GRAPH DATA LOADER (kèm edge_weight)
# ==============================================================================
def prepare_dual_graph_data(pkl_path: str,
                             orig_tsv_path: str,
                             target_tsv_path: str,
                             device: str = 'cpu'):
    """
    Load, lọc và chuẩn bị dữ liệu đồ thị kép Normal / Tumor.

    [U2] Tự động tạo edge_weight Uniform(0.5, 1.0) nếu file .pkl
         không chứa trường 'edge_weight'. Thay thế bằng trọng số thực
         (ví dụ HumanNet confidence score) để có kết quả tốt nhất.

    Returns:
        x_n        : Tensor [N, 14]   — features Normal  (Z-score)
        e_n        : Tensor [2, E_n]  — edge index Normal
        ew_n       : Tensor [E_n]     — edge weight Normal
        x_t        : Tensor [N, 14]   — features Tumor   (Z-score)
        e_t        : Tensor [2, E_t]  — edge index Tumor  (clone riêng)
        ew_t       : Tensor [E_t]     — edge weight Tumor
        valid_genes: list[str]        — danh sách gen theo thứ tự hàng
    """
    print("--- Đang nạp và lọc dữ liệu đồ thị ---")

    # ── 1. Đọc danh sách gen ────────────────────────────────────────────────
    orig_df    = pd.read_csv(orig_tsv_path, sep='\t')
    orig_genes = [str(g).strip().upper() for g in orig_df.iloc[:, 0].tolist()]

    target_df    = pd.read_csv(target_tsv_path, sep='\t')
    target_genes = set(str(g).strip().upper() for g in target_df.iloc[:, 0].tolist())

    # ── 2. Mapping old_idx -> new_idx ────────────────────────────────────────
    old_to_new_idx: dict = {}
    valid_genes: list    = []
    new_idx = 0
    for old_idx, gene in enumerate(orig_genes):
        if gene in target_genes:
            old_to_new_idx[old_idx] = new_idx
            valid_genes.append(gene)
            new_idx += 1
    print(f"[*] Khớp được {len(valid_genes)} gen vào đồ thị gốc.")

    # ── 3. Đọc PKL, cắt features, chuẩn hoá Z-score ─────────────────────────
    data           = pd.read_pickle(pkl_path)
    old_norm       = data['subtype_x']['Normal'].values    # [11k, 14]
    old_tumor      = data['subtype_x']['Tumor'].values     # [11k, 14]
    old_edge_index = np.array(data['edge_index'])          # [2, E_orig]

    # Đọc edge_weight thực nếu có, nếu không sẽ sinh dummy ở bước 4
    raw_edge_weight = data.get('edge_weight', None)        # None hoặc array [E_orig]

    keep_old = list(old_to_new_idx.keys())
    new_norm  = old_norm[keep_old]     # [N, 14]
    new_tumor = old_tumor[keep_old]    # [N, 14]

    scaler        = StandardScaler()
    new_norm_sc   = scaler.fit_transform(new_norm)
    new_tumor_sc  = scaler.transform(new_tumor)

    x_n = torch.tensor(new_norm_sc,  dtype=torch.float32).to(device)
    x_t = torch.tensor(new_tumor_sc, dtype=torch.float32).to(device)

    # ── 4. Lọc edges + build edge_weight ────────────────────────────────────
    new_edges   = []
    new_weights = []

    for i in range(old_edge_index.shape[1]):
        src, dst = int(old_edge_index[0, i]), int(old_edge_index[1, i])
        if src in old_to_new_idx and dst in old_to_new_idx:
            new_edges.append([old_to_new_idx[src], old_to_new_idx[dst]])
            if raw_edge_weight is not None:
                new_weights.append(float(raw_edge_weight[i]))
            else:
                # Dummy Uniform(0.5, 1.0) — thay bằng HumanNet score khi có
                new_weights.append(np.random.uniform(0.5, 1.0))

    if len(new_edges) > 0:
        new_edge_index  = np.array(new_edges,   dtype=np.int64).T   # [2, E_new]
        new_edge_weight = np.array(new_weights, dtype=np.float32)   # [E_new]
    else:
        new_edge_index  = np.empty((2, 0), dtype=np.int64)
        new_edge_weight = np.empty((0,),   dtype=np.float32)

    # Chuẩn hóa edge_weight về [0, 1] để ổn định huấn luyện
    if new_edge_weight.size > 0:
        w_min, w_max = new_edge_weight.min(), new_edge_weight.max()
        if w_max > w_min:
            new_edge_weight = (new_edge_weight - w_min) / (w_max - w_min + 1e-8)
        else:
            new_edge_weight = np.ones_like(new_edge_weight)

    if raw_edge_weight is None:
        print(f"[U2] Không tìm thấy edge_weight trong pkl → sinh dummy Uniform(0.5,1.0) "
              f"cho {new_edge_weight.shape[0]} cạnh.")
    else:
        print(f"[U2] Đọc edge_weight thực từ pkl ({new_edge_weight.shape[0]} cạnh).")

    print(f"[*] Edges: {old_edge_index.shape[1]} → {new_edge_index.shape[1]} sau khi lọc.")

    e_index  = torch.tensor(new_edge_index,  dtype=torch.long).to(device)
    ew_index = torch.tensor(new_edge_weight, dtype=torch.float32).to(device)

    # Tách biến e_n/e_t và ew_n/ew_t để sẵn sàng multiplex graph
    e_n  = e_index
    e_t  = e_index.clone()
    ew_n = ew_index
    ew_t = ew_index.clone()

    return x_n, e_n, ew_n, x_t, e_t, ew_t, valid_genes


# ==============================================================================
# GROUND TRUTH LOADER + TRANSDUCTIVE SPLIT (70 / 15 / 15)
# ==============================================================================
def load_and_split_ground_truth(ncg_path: str,
                                 oncokb_path: str,
                                 current_gene_list: list,
                                 train_ratio: float = 0.70,
                                 val_ratio:   float = 0.15,
                                 seed: int          = 42,
                                 device: str        = 'cpu'):
    """
    Load NCG + OncoKB, chia Transductive Split 70/15/15.

    Returns (tất cả là Tensor trên device):
        Y_train   [N]  — 1 ở train_pos, 0 ở phần còn lại (Unlabeled)
        Y_val     [N]  — Y_full (nhãn thật để chấm Val)
        Y_test    [N]  — Y_full (nhãn thật để chấm Test)
        Y_full    [N]  — toàn bộ nhãn thật
        val_mask  [N]  — Bool: True = được phép evaluate (loại train_pos)
        test_mask [N]  — Bool: True = được phép test  (loại train+val pos)
    """
    print("--- Đang nạp và phân chia Ground Truth (NCG & OncoKB) ---")
    driver_genes: set = set()

    for path, label in [(ncg_path, "NCG"), (oncokb_path, "OncoKB")]:
        try:
            df    = pd.read_csv(path, sep='\t')
            genes = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
            driver_genes.update(genes)
        except Exception as e:
            print(f"[Cảnh báo] Lỗi đọc {label}: {e}")

    print(f"[*] Tổng gen ung thư trong NCG + OncoKB: {len(driver_genes)}")

    Y_full = np.zeros(len(current_gene_list), dtype=np.float32)
    for i, gene in enumerate(current_gene_list):
        if gene in driver_genes:
            Y_full[i] = 1.0

    match_count = int(Y_full.sum())
    print(f"[*] Khớp {match_count} gen ung thư / {len(current_gene_list)} gen trong mạng.")

    # ── Transductive Split ───────────────────────────────────────────────────
    np.random.seed(seed)
    pos_indices = np.where(Y_full == 1.0)[0]
    np.random.shuffle(pos_indices)

    n_total = len(pos_indices)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_pos_idx = pos_indices[:n_train]
    val_pos_idx   = pos_indices[n_train: n_train + n_val]
    # test_pos_idx  = pos_indices[n_train + n_val:]   (dùng để log)
    test_pos_idx  = pos_indices[n_train + n_val:]

    print(f"[*] Train Positive : {len(train_pos_idx)}  gen  (nnPU học)")
    print(f"[*] Val   Positive : {len(val_pos_idx)}   gen  (Early Stopping)")
    print(f"[*] Test  Positive : {len(test_pos_idx)}  gen  (Đánh giá cuối)")

    Y_train                  = np.zeros_like(Y_full)
    Y_train[train_pos_idx]   = 1.0

    N         = len(current_gene_list)
    val_mask  = np.ones(N, dtype=bool)
    val_mask[train_pos_idx]  = False

    test_mask = np.ones(N, dtype=bool)
    test_mask[train_pos_idx] = False
    test_mask[val_pos_idx]   = False

    def tt(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).to(device)

    return (
        tt(Y_train),
        tt(Y_full),          # Y_val  ≡ Y_full
        tt(Y_full),          # Y_test ≡ Y_full
        tt(Y_full),
        tt(val_mask,  dtype=torch.bool),
        tt(test_mask, dtype=torch.bool),
    )