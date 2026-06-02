import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import random
import argparse
import gc
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# SEED
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# EARLY STOPPING (dựa trên Validation AUPRC - càng cao càng tốt)
# ==============================================================================
class EarlyStopping:
    """
    Dừng sớm khi Validation AUPRC không cải thiện sau `patience` epochs.
    Lưu lại bộ trọng số tốt nhất.
    """
    def __init__(self, patience=50, min_delta=1e-5):
        self.patience     = patience
        self.min_delta    = min_delta
        self.counter      = 0
        self.best_score   = -float('inf')   # Theo dõi AUPRC (cao hơn = tốt hơn)
        self.early_stop   = False
        self.best_weights = None

    def __call__(self, val_auprc, model):
        """
        Returns:
            True  -> đây là AUPRC tốt nhất, đã lưu weights
            False -> không cải thiện
        """
        if val_auprc > self.best_score + self.min_delta:
            self.best_score   = val_auprc
            self.counter      = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ==============================================================================
# HÀM EVALUATE (dùng cho cả Val và Test)
# ==============================================================================
@torch.no_grad()
def evaluate(model, x_n, x_t, e_n, e_t, Y_true, mask, device):
    """
    Tính ROC-AUC và AUPRC trên tập được chỉ định bởi mask.

    Args:
        Y_true : Tensor [N] - nhãn thật đầy đủ (Y_full)
        mask   : BoolTensor [N] - True = gen thuộc tập cần đánh giá

    Returns:
        (roc_auc, auprc) hoặc (0.0, 0.0) nếu không đủ class
    """
    model.eval()
    with torch.amp.autocast('cuda'):
        logits = model(x_n, x_t, e_n, e_t)
        probs  = torch.sigmoid(logits).cpu().numpy()

    mask_np = mask.cpu().numpy()
    y_true  = Y_true.cpu().numpy()[mask_np]
    y_prob  = probs[mask_np]

    # Kiểm tra đủ 2 class để tính metric
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    roc_auc = roc_auc_score(y_true, y_prob)
    auprc   = average_precision_score(y_true, y_prob)
    return roc_auc, auprc


# ==============================================================================
# MAIN
# ==============================================================================
def main(cancer_type, target_num_genes):
    seed_everything(42)

    parser = argparse.ArgumentParser(description="SiamesePUGNN - Transductive PU Learning")
    parser.add_argument('--base_data_dir', type=str, default='/content/drive/MyDrive/DATN/Data')
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--prior',         type=float, default=0.05,
                        help='Tỷ lệ gen ung thư ước tính (fix cứng theo kiến thức y sinh)')

    args, _ = parser.parse_known_args()

    # Import sau seed để tránh ảnh hưởng random state
    from utils    import prepare_dual_graph_data, load_and_split_ground_truth
    from modelVIP import SiamesePUGNN, robust_nnpu_loss

    save_dir = '/content/drive/MyDrive/DATN/Checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Mô hình SiamesePUGNN | Bệnh: {cancer_type} | Thiết bị: {device} ---")

    # -------------------------------------------------------------------------
    # 1. ĐƯỜNG DẪN DỮ LIỆU
    # -------------------------------------------------------------------------
    cancer_dir      = os.path.join(args.base_data_dir, cancer_type)
    pkl_path        = os.path.join(cancer_dir, f'{cancer_type}_input_data_humannet.pkl')
    orig_tsv_path   = os.path.join(cancer_dir, f'{cancer_type}_gene_index_humannet.tsv')
    target_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_training_genes_{target_num_genes}.tsv')
    ncg_path        = os.path.join(cancer_dir, f'{cancer_type}_pos.tsv')
    oncokb_path     = os.path.join(cancer_dir, f'{cancer_type}_oncokb_biomarker_drug_associations.tsv')

    # -------------------------------------------------------------------------
    # 2. LOAD DỮ LIỆU ĐỒ THỊ
    # -------------------------------------------------------------------------
    x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv_path, device
    )

    # -------------------------------------------------------------------------
    # 3. LOAD VÀ CHIA NHÃN (Transductive 70% Train / 15% Val / 15% Test)
    # -------------------------------------------------------------------------
    Y_train, Y_val, Y_test, Y_full, val_mask, test_mask = load_and_split_ground_truth(
        ncg_path, oncokb_path, current_gene_list,
        train_ratio=0.70, val_ratio=0.15,
        seed=42, device=device
    )

    # Prior cố định theo tham số (không tính động để đúng bản chất y sinh)
    prior = args.prior
    print(f"[*] Prior (fix cứng): {prior}")

    # -------------------------------------------------------------------------
    # 4. KHỞI TẠO MÔ HÌNH
    # -------------------------------------------------------------------------
    model = SiamesePUGNN(
        in_dim=14, hidden_dim=64, heads=4,
        num_gat_layers=3, dropout=0.3, drop_edge_p=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau theo dõi Val AUPRC (mode='max' vì AUPRC cao hơn = tốt hơn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-6
    )

    # Dùng GradScaler chỉ trên CUDA
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    early_stopping = EarlyStopping(patience=50, min_delta=1e-5)

    # -------------------------------------------------------------------------
    # 5. VÒNG LẶP HUẤN LUYỆN
    # -------------------------------------------------------------------------
    print("\n================== BẮT ĐẦU HUẤN LUYỆN ====================")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_n, x_t, e_n, e_t)
            loss   = robust_nnpu_loss(logits, Y_train, prior=prior)

        scaler.scale(loss).backward()

        # Clip gradient để tránh bùng nổ (yêu cầu của thiết kế)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # --- Validate mỗi 5 epoch ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_roc, val_auprc = evaluate(
                model, x_n, x_t, e_n, e_t, Y_full, val_mask, device
            )

            # Cập nhật scheduler theo Val AUPRC
            scheduler.step(val_auprc)

            improved = early_stopping(val_auprc, model)
            marker   = " *** BEST ***" if improved else ""

            print(
                f"Epoch {epoch+1:04d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val ROC-AUC: {val_roc:.4f} | "
                f"Val AUPRC: {val_auprc:.4f}{marker}"
            )

            if early_stopping.early_stop:
                print(f"--> [EARLY STOP] Dừng tại Epoch {epoch+1} | Best Val AUPRC: {early_stopping.best_score:.4f}")
                break

    # -------------------------------------------------------------------------
    # 6. KHÔI PHỤC TRỌNG SỐ TỐT NHẤT VÀ LƯU FILE
    # -------------------------------------------------------------------------
    if early_stopping.best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in early_stopping.best_weights.items()})
        print(f"\n[+] Đã tải lại trọng số tốt nhất (Val AUPRC = {early_stopping.best_score:.4f})")

    model_filename = f'siamese_pugnn_{cancer_type}.pth'
    model_path     = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"[+] Đã lưu mô hình: {model_path}")

    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 7. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (CHỐNG DATA LEAKAGE)
    # -------------------------------------------------------------------------
    print(f"\n============== KẾT QUẢ CUỐI CÙNG - {cancer_type} ==============")

    # Val metrics (để đối chiếu)
    val_roc_final, val_auprc_final = evaluate(
        model, x_n, x_t, e_n, e_t, Y_full, val_mask, device
    )

    # Test metrics (tập hoàn toàn ẩn trong suốt quá trình train)
    test_roc, test_auprc = evaluate(
        model, x_n, x_t, e_n, e_t, Y_full, test_mask, device
    )

    print(f"{'Metric':<15} {'Validation':>12} {'Test':>12}")
    print("-" * 40)
    print(f"{'ROC-AUC':<15} {val_roc_final:>12.4f} {test_roc:>12.4f}")
    print(f"{'AUPRC':<15} {val_auprc_final:>12.4f} {test_auprc:>12.4f}  <- Quan trọng nhất")
    print("-" * 40)

    return {
        'cancer_type' : cancer_type,
        'val_roc'     : val_roc_final,
        'val_auprc'   : val_auprc_final,
        'test_roc'    : test_roc,
        'test_auprc'  : test_auprc,
    }


# ==============================================================================
# ==============================================================================
# HÀM BỔ TRỢ: TỰ ĐỘNG QUÉT THƯ MỤC
# ==============================================================================
def get_all_cancer_folders(base_dir):
    """Quét tất cả các thư mục bệnh trong thư mục Data, bỏ qua các thư mục nhiễu."""
    ignore_list = ['CPDB', 'STRING', '.ipynb_checkpoints', '__pycache__']
    if not os.path.exists(base_dir):
        return []
    
    folders = [f for f in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, f)) and f not in ignore_list]
    return sorted(folders)

# ==============================================================================
# CHẠY TỰ ĐỘNG HÀNG LOẠT (BATCH TRAINING)
# ==============================================================================
if __name__ == "__main__":
    BASE_DIR = '/content/drive/MyDrive/DATN/Data'
    NUM_GENES = 9000
    
    # -------------------------------------------------------------------------
    # CÁCH 1: TỰ QUÉT TOÀN BỘ CÁC BỆNH (Bỏ comment dòng dưới để dùng)
    # cancer_list = get_all_cancer_folders(BASE_DIR)
    
    # CÁCH 2: CHỈ ĐỊNH RÕ DANH SÁCH CÁC BỆNH MUỐN CHẠY
    # Bạn có thể thêm bớt tên thư mục bệnh vào mảng này
    cancer_list = ['BLCA', 'LUAD', 'BRCA', 'KIRC', 'KIRP','CESC','COAD','ESCA','HNSC','LIHC','LUSC','PRAD','STAD','THCA','UCEC'] 
    # -------------------------------------------------------------------------

    print("=" * 70)
    print(f"[*] HỆ THỐNG SẼ HUẤN LUYỆN LẦN LƯỢT {len(cancer_list)} BỆNH: {cancer_list}")
    print("=" * 70)

    # Danh sách lưu trữ thành tích để in bảng báo cáo cuối cùng
    all_results = []

    for cancer in cancer_list:
        print("\n" + "★" * 70)
        print(f"🚀 KHỞI ĐỘNG CHU TRÌNH CHO BỆNH: {cancer}")
        print("★" * 70)
        
        try:
            # Gọi hàm main() cho bệnh hiện tại
            res = main(cancer_type=cancer, target_num_genes=NUM_GENES)
            all_results.append(res)
            
        except Exception as e:
            # TRY-EXCEPT CỰC KỲ QUAN TRỌNG: 
            # Nếu 1 bệnh bị thiếu file hoặc lỗi data, nó sẽ báo lỗi rồi chạy tiếp bệnh sau
            # thay vì làm sập (crash) toàn bộ quá trình cắm máy qua đêm của bạn.
            print(f"\n[!] LỖI NGHIÊM TRỌNG Ở BỆNH {cancer}: {e}")
            print(f"-> Hệ thống bỏ qua {cancer} và tiếp tục với bệnh tiếp theo...\n")
            continue

    # ==============================================================================
    # IN BẢNG BÁO CÁO TỔNG KẾT CUỐI CÙNG
    # ==============================================================================
    print("\n" + "=" * 60)
    print(" BẢNG BÁO CÁO THÀNH TÍCH TỔNG HỢP (TEST SET)")
    print("=" * 60)
    print(f"{'BỆNH':<10} | {'ROC-AUC':<15} | {'AUPRC (Chính)':<15}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['cancer_type']:<10} | {r['test_roc']:<15.4f} | {r['test_auprc']:<15.4f}")
    
    print("=" * 60)
    print(" HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!")