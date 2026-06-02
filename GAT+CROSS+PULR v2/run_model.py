"""
run_model.py  —  Training & Evaluation Pipeline cho SiamesePUGNN v2
=====================================================================
Nâng cấp so với phiên bản trước:
  [R1] Multi-run (5 seeds): In Mean ± Std cho ROC-AUC và AUPRC
  [R2] Inductive placeholder: train_cancer_A → test_cancer_B
  [R3] AlphaMax prior: gọi estimate_prior_alphamax() sau khi load data
  [R4] Truyền edge_weight vào model.forward()
  [R5] Early Stopping theo Val AUPRC + ReduceLROnPlateau
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import warnings
import argparse
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# SEED
# ==============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==============================================================================
# EARLY STOPPING (theo Val AUPRC — cao hơn = tốt hơn)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-5):
        self.patience     = patience
        self.min_delta    = min_delta
        self.counter      = 0
        self.best_score   = -float('inf')
        self.early_stop   = False
        self.best_weights = None

    def __call__(self, val_auprc: float, model: torch.nn.Module) -> bool:
        """
        Returns True nếu là điểm tốt nhất và đã lưu weights.
        """
        if val_auprc > self.best_score + self.min_delta:
            self.best_score   = val_auprc
            self.counter      = 0
            self.best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# ==============================================================================
# EVALUATE
# ==============================================================================
@torch.no_grad()
def evaluate(model, x_n, x_t, e_n, ew_n, e_t, ew_t,
             Y_full, mask, device, use_amp: bool = True):
    """
    Tính ROC-AUC và AUPRC trên tập được chỉ định bởi mask.

    Returns: (roc_auc: float, auprc: float)
    """
    model.eval()
    with torch.amp.autocast('cuda', enabled=use_amp):
        logits = model(x_n, x_t, e_n, ew_n, e_t, ew_t)
        probs  = torch.sigmoid(logits).cpu().numpy()

    mask_np = mask.cpu().numpy()
    y_true  = Y_full.cpu().numpy()[mask_np]
    y_prob  = probs[mask_np]

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    return (
        float(roc_auc_score(y_true, y_prob)),
        float(average_precision_score(y_true, y_prob)),
    )


# ==============================================================================
# MỘT LẦN CHẠY (1 seed)
# ==============================================================================
def run_one_seed(seed: int,
                 cancer_type: str,
                 target_num_genes: int,
                 args,
                 device: torch.device,
                 save_dir: str) -> dict:
    """
    Thực hiện toàn bộ pipeline train + eval cho 1 seed.

    Returns:
        dict với các key: val_roc, val_auprc, test_roc, test_auprc
    """
    from utils import (prepare_dual_graph_data,
                       load_and_split_ground_truth,
                       estimate_prior_alphamax)
    from model import SiamesePUGNN, robust_nnpu_loss

    seed_everything(seed)
    print(f"\n{'='*60}")
    print(f"  SEED {seed}  |  {cancer_type}  |  {device}")
    print(f"{'='*60}")

    # ── Đường dẫn ──────────────────────────────────────────────────────────
    cancer_dir      = os.path.join(args.base_data_dir, cancer_type)
    pkl_path        = os.path.join(cancer_dir, f'{cancer_type}_input_data_humannet.pkl')
    orig_tsv_path   = os.path.join(cancer_dir, f'{cancer_type}_gene_index_humannet.tsv')
    target_tsv_path = os.path.join(cancer_dir,
                                   f'{cancer_type}_training_genes_{target_num_genes}.tsv')
    ncg_path        = os.path.join(cancer_dir, f'{cancer_type}_pos.tsv')
    oncokb_path     = os.path.join(cancer_dir,
                                   f'{cancer_type}_oncokb_biomarker_drug_associations.tsv')

    # ── Load đồ thị ─────────────────────────────────────────────────────────
    x_n, e_n, ew_n, x_t, e_t, ew_t, gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv_path, device
    )

    # ── Load nhãn ───────────────────────────────────────────────────────────
    Y_train, _, _, Y_full, val_mask, test_mask = load_and_split_ground_truth(
        ncg_path, oncokb_path, gene_list,
        train_ratio=0.70, val_ratio=0.15,
        seed=seed, device=device,
    )

    # ── [R3] AlphaMax prior ─────────────────────────────────────────────────
    prior = estimate_prior_alphamax(x_n, x_t, Y_train,
                                    clip_low=0.01, clip_high=0.15)
    print(f"[*] Prior (AlphaMax, seed={seed}): {prior:.4f}")

    # ── Khởi tạo model ──────────────────────────────────────────────────────
    use_amp = (device.type == 'cuda')
    model   = SiamesePUGNN(
        in_dim=14, hidden_dim=64, heads=4,
        num_gat_layers=3, dropout=0.3, drop_edge_p=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20,
        min_lr=1e-6,
    )
    scaler        = torch.amp.GradScaler('cuda', enabled=use_amp)
    early_stopper = EarlyStopping(patience=50, min_delta=1e-5)

    # ── Vòng lặp huấn luyện ──────────────────────────────────────────────────
    print(f"\n--- Bắt đầu huấn luyện (max {args.epochs} epochs) ---")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_n, x_t, e_n, ew_n, e_t, ew_t)
            loss   = robust_nnpu_loss(logits, Y_train, prior=prior)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Validate mỗi 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_roc, val_auprc = evaluate(
                model, x_n, x_t, e_n, ew_n, e_t, ew_t,
                Y_full, val_mask, device, use_amp,
            )
            scheduler.step(val_auprc)
            improved = early_stopper(val_auprc, model)
            mark = " ★ BEST" if improved else ""
            print(
                f"  Epoch {epoch+1:04d}/{args.epochs} | "
                f"Loss={loss.item():.4f} | "
                f"Val AUC={val_roc:.4f} | Val AUPRC={val_auprc:.4f}{mark}"
            )

            if early_stopper.early_stop:
                print(f"  → Early Stop @ epoch {epoch+1} | "
                      f"Best Val AUPRC={early_stopper.best_score:.4f}")
                break

    # ── Khôi phục best weights ──────────────────────────────────────────────
    if early_stopper.best_weights:
        model.load_state_dict(
            {k: v.to(device) for k, v in early_stopper.best_weights.items()}
        )

    # ── Lưu checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(save_dir,
                             f'siamese_pugnn_{cancer_type}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [+] Đã lưu checkpoint: {ckpt_path}")

    # ── Đánh giá cuối ───────────────────────────────────────────────────────
    val_roc_f,  val_auprc_f  = evaluate(
        model, x_n, x_t, e_n, ew_n, e_t, ew_t,
        Y_full, val_mask, device, use_amp,
    )
    test_roc_f, test_auprc_f = evaluate(
        model, x_n, x_t, e_n, ew_n, e_t, ew_t,
        Y_full, test_mask, device, use_amp,
    )
    print(f"\n  [Seed {seed}] Val  | AUC={val_roc_f:.4f}  AUPRC={val_auprc_f:.4f}")
    print(f"  [Seed {seed}] Test | AUC={test_roc_f:.4f}  AUPRC={test_auprc_f:.4f}")

    torch.cuda.empty_cache()
    gc.collect()

    return {
        'val_roc':    val_roc_f,
        'val_auprc':  val_auprc_f,
        'test_roc':   test_roc_f,
        'test_auprc': test_auprc_f,
    }


# ==============================================================================
# [R1]  MULTI-RUN: 5 SEEDS
# ==============================================================================
def run_multirun(cancer_type: str,
                 target_num_genes: int,
                 args,
                 seeds: list = None):
    """
    Chạy pipeline 5 lần với 5 seeds khác nhau, in Mean ± Std cuối cùng.
    """
    if seeds is None:
        seeds = [42, 69, 123, 456, 999]

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = []
    for seed in seeds:
        result = run_one_seed(
            seed=seed,
            cancer_type=cancer_type,
            target_num_genes=target_num_genes,
            args=args,
            device=device,
            save_dir=save_dir,
        )
        all_results.append(result)

    # ── Tổng hợp thống kê ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT MULTI-RUN  |  {cancer_type}  |  {len(seeds)} seeds")
    print(f"{'='*60}")

    for metric in ['val_roc', 'val_auprc', 'test_roc', 'test_auprc']:
        vals = np.array([r[metric] for r in all_results])
        print(f"  {metric:15s}: {vals.mean():.4f} ± {vals.std():.4f}  "
              f"(seeds={seeds})")

    # Trả về dict để dùng ngoài nếu cần
    return all_results


# ==============================================================================
# [R2]  INDUCTIVE PLACEHOLDER: Train trên bệnh A → Test trên bệnh B
# ==============================================================================
def run_inductive(train_cancer:      str,
                  test_cancer:       str,
                  target_num_genes:  int,
                  args,
                  seed: int = 42):
    """
    Inductive Setup:
        1. Train đầy đủ trên đồ thị của train_cancer.
        2. Load trọng số đã lưu.
        3. Chạy inference (không finetune) trên đồ thị của test_cancer.

    Điều kiện để hoạt động:
        • Cả 2 bệnh phải có cùng in_dim (14) — đã thỏa mãn.
        • Số lượng gen (N) có thể KHÁC nhau vì model không có
          weight cố định theo N (chỉ GAT + Linear).

    Args:
        train_cancer     : Ví dụ 'BLCA'
        test_cancer      : Ví dụ 'LUAD'
        target_num_genes : Số gen lọc (dùng chung cho 2 bệnh)
        args             : argparse Namespace
        seed             : seed cho data split
    """
    from utils import (prepare_dual_graph_data,
                       load_and_split_ground_truth,
                       estimate_prior_alphamax)
    from model import SiamesePUGNN, robust_nnpu_loss

    seed_everything(seed)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── BƯỚC 1: Huấn luyện trên bệnh A ──────────────────────────────────────
    print(f"\n[Inductive] === Huấn luyện trên {train_cancer} ===")
    train_dir       = os.path.join(args.base_data_dir, train_cancer)
    pkl_A           = os.path.join(train_dir, f'{train_cancer}_input_data_humannet.pkl')
    orig_tsv_A      = os.path.join(train_dir, f'{train_cancer}_gene_index_humannet.tsv')
    target_tsv_A    = os.path.join(train_dir,
                                   f'{train_cancer}_training_genes_{target_num_genes}.tsv')
    ncg_A           = os.path.join(train_dir, f'{train_cancer}_pos.tsv')
    oncokb_A        = os.path.join(train_dir,
                                   f'{train_cancer}_oncokb_biomarker_drug_associations.tsv')

    x_n_A, e_n_A, ew_n_A, x_t_A, e_t_A, ew_t_A, genes_A = prepare_dual_graph_data(
        pkl_A, orig_tsv_A, target_tsv_A, device
    )
    Y_train_A, _, _, Y_full_A, val_mask_A, test_mask_A = load_and_split_ground_truth(
        ncg_A, oncokb_A, genes_A,
        train_ratio=0.70, val_ratio=0.15, seed=seed, device=device,
    )
    prior_A = estimate_prior_alphamax(x_n_A, x_t_A, Y_train_A)
    print(f"[Inductive] Prior {train_cancer} (AlphaMax): {prior_A:.4f}")

    use_amp = (device.type == 'cuda')
    model   = SiamesePUGNN(in_dim=14, hidden_dim=64, heads=4,
                           num_gat_layers=3, dropout=0.3).to(device)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler        = torch.amp.GradScaler('cuda', enabled=use_amp)
    early_stopper = EarlyStopping(patience=50)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-6, verbose=False
    )

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_n_A, x_t_A, e_n_A, ew_n_A, e_t_A, ew_t_A)
            loss   = robust_nnpu_loss(logits, Y_train_A, prior=prior_A)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_roc, val_auprc = evaluate(
                model, x_n_A, x_t_A, e_n_A, ew_n_A, e_t_A, ew_t_A,
                Y_full_A, val_mask_A, device, use_amp,
            )
            scheduler.step(val_auprc)
            if early_stopper(val_auprc, model):
                print(f"  Epoch {epoch+1:04d} | Loss={loss.item():.4f} | "
                      f"Val AUPRC={val_auprc:.4f} ★")
            if early_stopper.early_stop:
                print(f"  → Early Stop @ epoch {epoch+1}")
                break

    if early_stopper.best_weights:
        model.load_state_dict(
            {k: v.to(device) for k, v in early_stopper.best_weights.items()}
        )

    ckpt_path = os.path.join(save_dir, f'inductive_{train_cancer}_seed{seed}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Inductive] Đã lưu model {train_cancer}: {ckpt_path}")

    # ── BƯỚC 2: Inference trên bệnh B (KHÔNG finetune) ───────────────────────
    print(f"\n[Inductive] === Inference trên {test_cancer} (zero-shot) ===")
    test_dir     = os.path.join(args.base_data_dir, test_cancer)
    pkl_B        = os.path.join(test_dir, f'{test_cancer}_input_data_humannet.pkl')
    orig_tsv_B   = os.path.join(test_dir, f'{test_cancer}_gene_index_humannet.tsv')
    target_tsv_B = os.path.join(test_dir,
                                f'{test_cancer}_training_genes_{target_num_genes}.tsv')
    ncg_B        = os.path.join(test_dir, f'{test_cancer}_pos.tsv')
    oncokb_B     = os.path.join(test_dir,
                                f'{test_cancer}_oncokb_biomarker_drug_associations.tsv')

    x_n_B, e_n_B, ew_n_B, x_t_B, e_t_B, ew_t_B, genes_B = prepare_dual_graph_data(
        pkl_B, orig_tsv_B, target_tsv_B, device
    )
    _, _, _, Y_full_B, _, test_mask_B = load_and_split_ground_truth(
        ncg_B, oncokb_B, genes_B,
        train_ratio=0.70, val_ratio=0.15, seed=seed, device=device,
    )

    # Load lại checkpoint bệnh A vào model mới (đề phòng N khác nhau → GAT vẫn OK)
    model_B = SiamesePUGNN(in_dim=14, hidden_dim=64, heads=4,
                            num_gat_layers=3, dropout=0.3).to(device)
    model_B.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_roc, test_auprc = evaluate(
        model_B, x_n_B, x_t_B, e_n_B, ew_n_B, e_t_B, ew_t_B,
        Y_full_B, test_mask_B, device, use_amp,
    )
    print(f"\n[Inductive] Train={train_cancer} → Test={test_cancer}")
    print(f"  ROC-AUC : {test_roc:.4f}")
    print(f"  AUPRC   : {test_auprc:.4f}  ← Quan trọng nhất")

    torch.cuda.empty_cache()
    gc.collect()
    return {'test_roc': test_roc, 'test_auprc': test_auprc}


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(description='SiamesePUGNN v2 — run_model.py')
    parser.add_argument('--base_data_dir', type=str,
                        default='/content/drive/MyDrive/DATN/Data')
    parser.add_argument('--save_dir',      type=str,
                        default='/content/drive/MyDrive/DATN/Checkpoints')
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--mode',          type=str,
                        choices=['multirun', 'inductive', 'single'],
                        default='multirun',
                        help='multirun=5 seeds | inductive=A→B | single=1 seed')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='Seed dùng cho mode=single hoặc mode=inductive')
    parser.add_argument('--train_cancer',  type=str,   default='BLCA',
                        help='Bệnh huấn luyện (dùng cho mode=inductive)')
    parser.add_argument('--test_cancer',   type=str,   default='LUAD',
                        help='Bệnh kiểm tra  (dùng cho mode=inductive)')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    # ── CẤU HÌNH BATCH TRAINING (CHẠY NHIỀU BỆNH) ───────────────────────────
    # Bạn thêm hoặc bớt tên các bệnh vào danh sách này:
    CANCER_TYPES = ['THCA', 'HNSC','LIHC','LUSC','STAD','UCEC'] 
    NUM_GENES    = 9000
    # ─────────────────────────────────────────────────────────────────────────

    args = get_args()
    
    # Danh sách để lưu lại kết quả in ra bảng cuối cùng
    summary_table = []

    print("=" * 70)
    print(f"[*] BẮT ĐẦU CHẠY HÀNG LOẠT {len(CANCER_TYPES)} BỆNH:")
    print(f"    {CANCER_TYPES}")
    print(f"[*] CHẾ ĐỘ: {args.mode.upper()}")
    print("=" * 70)

    for CANCER_TYPE in CANCER_TYPES:
        print("\n" + "★" * 70)
        print(f"🚀 ĐANG XỬ LÝ BỆNH: {CANCER_TYPE} 🚀")
        print("★" * 70)

        try:
            if args.mode == 'multirun':
                # [R1] Chạy nhiều seeds (Mặc định 5 seeds như code gốc)
                seeds_list = [42, 69, 123, 456, 999]
                results = run_multirun(
                    cancer_type      = CANCER_TYPE,
                    target_num_genes = NUM_GENES,
                    args             = args,
                    seeds            = seeds_list,
                )
                
                # Tính trung bình các seed để đưa vào bảng tổng kết
                avg_roc   = sum(r['test_roc'] for r in results) / len(results)
                avg_auprc = sum(r['test_auprc'] for r in results) / len(results)
                
                summary_table.append({
                    'cancer': CANCER_TYPE,
                    'roc': avg_roc,
                    'auprc': avg_auprc,
                    'note': f"Mean ({len(seeds_list)} seeds)"
                })

            elif args.mode == 'inductive':
                # [R2] Train trên CANCER_TYPE hiện tại, test trên args.test_cancer
                result = run_inductive(
                    train_cancer     = CANCER_TYPE,      # Lấy lần lượt từ danh sách
                    test_cancer      = args.test_cancer, # Giữ nguyên bệnh test
                    target_num_genes = NUM_GENES,
                    args             = args,
                    seed             = args.seed,
                )
                summary_table.append({
                    'cancer': f"{CANCER_TYPE} → {args.test_cancer}",
                    'roc': result['test_roc'],
                    'auprc': result['test_auprc'],
                    'note': "Inductive"
                })

            elif args.mode == 'single':
                # [R3] Chạy 1 seed duy nhất
                device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                save_dir = args.save_dir
                os.makedirs(save_dir, exist_ok=True)
                result = run_one_seed(
                    seed             = args.seed,
                    cancer_type      = CANCER_TYPE,
                    target_num_genes = NUM_GENES,
                    args             = args,
                    device           = device,
                    save_dir         = save_dir,
                )
                print(f"\n[{CANCER_TYPE} - Single] Test AUC={result['test_roc']:.4f} | "
                      f"AUPRC={result['test_auprc']:.4f}")
                
                summary_table.append({
                    'cancer': CANCER_TYPE,
                    'roc': result['test_roc'],
                    'auprc': result['test_auprc'],
                    'note': f"Seed {args.seed}"
                })

        except Exception as e:
            # Rất quan trọng: Bắt lỗi để không bị sập toàn bộ tiến trình
            print(f"\n[!] CẢNH BÁO: Xảy ra lỗi khi chạy bệnh {CANCER_TYPE}: {e}")
            print(f"-> Hệ thống sẽ bỏ qua {CANCER_TYPE} và chạy tiếp bệnh sau.\n")
            continue

    # ==============================================================================
    # IN BẢNG BÁO CÁO TỔNG KẾT CUỐI CÙNG
    # ==============================================================================
    print("\n" + "=" * 70)
    print("🏆 BẢNG TỔNG KẾT KẾT QUẢ HUẤN LUYỆN (TEST SET) 🏆")
    print("=" * 70)
    print(f"{'BỆNH':<15} | {'ROC-AUC':<12} | {'AUPRC':<12} | {'GHI CHÚ':<20}")
    print("-" * 70)
    
    for row in summary_table:
        print(f"{row['cancer']:<15} | {row['roc']:<12.4f} | {row['auprc']:<12.4f} | {row['note']:<20}")
    
    print("=" * 70)
    print("🎉 ĐÃ HOÀN TẤT TOÀN BỘ DANH SÁCH!")