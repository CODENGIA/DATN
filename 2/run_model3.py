import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import argparse
import gc
import warnings
import copy
from scipy.stats import rankdata
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_recall_curve, auc)

warnings.filterwarnings("ignore", category=FutureWarning)

from utils import prepare_dual_graph_data, load_ground_truth_labels
from model import MGAEDualGraphModel, mgae_loss, anomaly_score_per_node

def main():
    parser = argparse.ArgumentParser(
        description="Masked Graph AutoEncoder (MGAE) — Dual Graph Cancer Driver Detection"
    )
    parser.add_argument('--data_dir',    type=str,   default='/content/drive/MyDrive/DATN/Data/BLCA')
    parser.add_argument('--cancer_type', type=str,   default='BLCA',
                        help='Tên loại ung thư để tự động ghép tên file (BLCA, LUAD, ...)')
    parser.add_argument('--epochs',      type=int,   default=1000) # Đã tăng mặc định lên 1000
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--mask_ratio',  type=float, default=0.5,
                        help='Tỉ lệ node bị mask khi training (mặc định 0.5 = 50%)')
    parser.add_argument('--embed_dim',   type=int,   default=64)
    
    args = parser.parse_args(args=[])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang chạy mô hình trên: {device} ---")
    
    ct = args.cancer_type
    
    # =========================================================================
    # 1. NẠP DỮ LIỆU
    # =========================================================================
    pkl_path      = os.path.join(args.data_dir, f'{ct}_input_data_humannet.pkl')
    orig_tsv_path = os.path.join(args.data_dir, f'{ct}_gene_index_humannet.tsv')
    target_tsv    = os.path.join(args.data_dir, 'training_genes_6500.tsv')
    
    x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv, device
    )

    # =========================================================================
    # 2. NẠP GROUND TRUTH
    # =========================================================================
    ncg_path    = os.path.join(args.data_dir, f'{ct}_pos.tsv')
    oncokb_path = os.path.join(args.data_dir, f'{ct}_oncokb_biomarker_drug_associations.tsv')
    
    Y_test = load_ground_truth_labels(ncg_path, oncokb_path, current_gene_list, device)
    
    if Y_test.sum() == 0:
        print("\n[LỖI] Không tìm thấy gen bệnh nào trong Ground Truth!")
        return
        
    print(f"[*] Ground Truth OK — {int(Y_test.sum())} gen bệnh / {len(current_gene_list)} gen tổng")

    # =========================================================================
    # 3. KHỞI TẠO MÔ HÌNH VÀ BỘ TỐI ƯU
    # =========================================================================
    input_dim = x_n.shape[1]   # 14
    model = MGAEDualGraphModel(
        input_dim      = input_dim,
        embed_dim      = args.embed_dim,
        num_heads      = 4,
        decoder_hidden = 32,
        dropout        = 0.2,
        mask_ratio     = args.mask_ratio
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # BỔ SUNG: Scheduler tự động giảm Learning Rate nếu Loss không giảm trong 30 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    scaler    = torch.amp.GradScaler('cuda')
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Tổng tham số mô hình: {total_params:,}")

    # BỔ SUNG: Biến lưu trữ trọng số mô hình tốt nhất
    best_loss = float('inf')
    best_model_weights = None

    # =========================================================================
    # 4. TRAINING — CHỈ HỌC TẾ BÀO BÌNH THƯỜNG (NORMAL)
    # =========================================================================
    print(f"\n================ BẮT ĐẦU HUẤN LUYỆN MGAE ({args.epochs} epochs) ================")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            recon_n, recon_t, mask_n, mask_t = model.forward_train(x_n, e_n, x_t, e_t)
            
            # CHỈ TÍNH LOSS TRÊN NORMAL
            loss_n = mgae_loss(x_n, recon_n, mask_n)
            loss   = loss_n
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Cập nhật Scheduler
        scheduler.step(loss)
        
        # Cập nhật mô hình tốt nhất
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_weights = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:04d}/{args.epochs} | LR: {current_lr:.6f} | Train Normal Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")
                  
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 5. INFERENCE — TÍNH ĐỘ LỆCH SAI SỐ (TUMOR trừ NORMAL)
    # =========================================================================
    print("\n================ ĐÁNH GIÁ BẰNG RECONSTRUCTION ERROR (MGAE) ================")
    # BỔ SUNG: Nạp lại bộ trọng số tốt nhất trước khi test
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"[*] Đã nạp lại trọng số tại Epoch có Loss thấp nhất: {best_loss:.4f}")

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            recon_n_full, recon_t_full = model.forward_inference(x_n, e_n, x_t, e_t)
            
        # MSE per node cho từng đồ thị
        err_n = anomaly_score_per_node(x_n, recon_n_full)
        err_t = anomaly_score_per_node(x_t, recon_t_full)
        
        # Điểm dị thường = Sai số Tumor TRỪ ĐI Sai số Normal
        err_combined = err_t - err_n
        
        # Rank normalization
        anomaly_scores = rankdata(err_combined) / len(err_combined)
        
    y_test_np = Y_test.cpu().numpy()
    
    # Tính metrics
    precisions, recalls, thresholds = precision_recall_curve(y_test_np, anomaly_scores)
    auprc      = auc(recalls, precisions)
    f1_arr     = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thr   = thresholds[np.argmax(f1_arr)]
    preds_np   = (anomaly_scores > best_thr).astype(float)
    
    auc_score  = roc_auc_score(y_test_np, anomaly_scores)
    aupr       = average_precision_score(y_test_np, anomaly_scores)
    f1         = f1_score(y_test_np, preds_np)
    
    print(f"[*] Ngưỡng tối ưu F1: {best_thr:.4f}")
    print(f"\n>>> KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ (MGAE) <<<")
    print(f"AUC    = {auc_score:.4f}")
    print(f"AUPR   = {aupr:.4f}")
    print(f"AUPRC  = {auprc:.4f}")
    print(f"F1     = {f1:.4f}")

if __name__ == "__main__":
    main()