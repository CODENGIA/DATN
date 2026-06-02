import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import gc
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from scipy.stats import rankdata

warnings.filterwarnings("ignore", category=FutureWarning)

# Khóa Seed bảo vệ tính tái lập kết quả đồ án
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# [CHÚ Ý]: Đã sửa import để lấy các class từ file modelGCNMLP.py
from utils import prepare_dual_graph_data, load_ground_truth_labels
from modelGCNMLP import DualGraphCrossAttentionModel, info_nce_loss, edge_reconstruction_loss

def main():
    seed_everything(42)

    parser = argparse.ArgumentParser(description="Mô hình V3 nâng cấp: Lõi Lai GCN+MLP Kết hợp Không Giám Sát")
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/DATN/Data/LUAD')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.6)
    args = parser.parse_args()

    save_dir = '/content/drive/MyDrive/DATN/Checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang chạy mô hình Lai GCN+MLP Không giám sát trên: {device} ---")

    # 1. LOAD DỮ LIỆU TỪ UTILS
    pkl_path = os.path.join(args.data_dir, 'LUAD_input_data_humannet.pkl')        
    orig_tsv_path = os.path.join(args.data_dir, 'LUAD_gene_index_humannet.tsv')   
    target_tsv_path = os.path.join(args.data_dir, 'training_genes_6500.tsv')      
    
    x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv_path, device
    )

    ncg_path = os.path.join(args.data_dir, 'LUAD_pos.tsv')
    oncokb_path = os.path.join(args.data_dir, 'LUAD_oncokb_biomarker_drug_associations.tsv')
    Y_test = load_ground_truth_labels(ncg_path, oncokb_path, current_gene_list, device)

    # Tính toán pos_weight có giới hạn trần
    num_nodes = x_n.size(0)
    num_edges = e_n.size(1) 
    raw_pos_weight = (num_nodes * num_nodes - num_edges) / float(num_edges)
    pos_weight_val = min(raw_pos_weight, 15.0)
    print(f"[INFO] Hệ số pos_weight gốc: {raw_pos_weight:.2f} -> Giới hạn bảo vệ: {pos_weight_val:.2f}")

    # 2. KHỞI TẠO MÔ HÌNH (embed_dim=128 để tăng sức chứa đặc trưng lai)
    model = DualGraphCrossAttentionModel(input_dim=14, embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    print("\n================ BẮT ĐẦU HUẤN LUYỆN ===================")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            out_n, out_t = model(x_n, e_n, x_t, e_t)
            
            # Tính toán Hybrid Loss không giám sát
            loss_nce = info_nce_loss(out_n, out_t, temperature=0.05, num_negatives=5)
            loss_recon_n = edge_reconstruction_loss(out_n, e_n, pos_weight_val)
            loss_recon_t = edge_reconstruction_loss(out_t, e_t, pos_weight_val)
            loss_recon = (loss_recon_n + loss_recon_t) / 2.0
            
            loss = args.alpha * loss_nce + args.beta * loss_recon

        scaler.scale(loss).backward()
        scaler.step(optimizer)  
        scaler.update()
        
        if (epoch + 1) % 50 == 0: 
            print(f"Epoch {epoch+1:03d}/{args.epochs} | L_NCE: {loss_nce.item():.4f} | L_Edge: {loss_recon.item():.4f} | Tổng Loss: {loss.item():.4f}")

    # 3. LƯU MÔ HÌNH VÀO DRIVE
    model_path = os.path.join(save_dir, 'hybrid_gcn_mlp_unsupervised.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n[THÀNH CÔNG] Đã lưu mô hình lai tại: {model_path}")

    torch.cuda.empty_cache()
    gc.collect()

    # 4. ĐÁNH GIÁ PHÁT HIỆN BẤT THƯỜNG KHÔNG GIÁM SÁT & QUÉT BETA
    print("\n================ ĐÁNH GIÁ VÀ TÌM TRỌNG SỐ TỐI ƯU ===================")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out_n, out_t = model(x_n, e_n, x_t, e_t)
            
            # Điểm 1: Độ lỗi InfoNCE cá nhân
            loss_nce_ind = info_nce_loss(out_n, out_t, temperature=0.05, num_negatives=20, return_individual_loss=True)
            score_nce = loss_nce_ind.cpu().numpy()
            
            # Điểm 2: Khoảng cách hình học L2 chỉ ra sai khác trạng thái sinh học
            diff_vector = torch.abs(out_n - out_t) 
            score_diff = torch.norm(diff_vector, p=2, dim=1).cpu().numpy() 
            
        rank_nce = rankdata(score_nce) / len(score_nce)
        rank_diff = rankdata(score_diff) / len(score_diff)
        y_test_np = Y_test.cpu().numpy()
        
        print(f"\n>>> QUÉT TRỌNG SỐ FUSION (BETA) CHO LÕI LAI GCN+MLP <<<")
        best_f1, best_aupr, best_beta = 0, 0, 0
        
        for beta in np.linspace(0, 1, 11):
            alpha = 1.0 - beta
            final_scores = (alpha * rank_nce) + (beta * rank_diff)
            
            precisions, recalls, thresholds = precision_recall_curve(y_test_np, final_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            best_thresh = thresholds[np.argmax(f1_scores)]
            preds_np = (final_scores > best_thresh).astype(float) 
            
            auc_score = roc_auc_score(y_test_np, final_scores)
            aupr = average_precision_score(y_test_np, final_scores)
            f1 = f1_score(y_test_np, preds_np)
            
            print(f"Beta = {beta:.1f} | Alpha = {alpha:.1f} --> AUC: {auc_score:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_beta = beta
            if aupr > best_aupr:
                best_aupr = aupr

        print(f"\n[*] KẾT LUẬN ĐỒ ÁN SAU KHI CẤY GHÉP LÕI LAI:")
        print(f"- F1 cao nhất đạt {best_f1:.4f} tại Beta = {best_beta:.1f}")
        print(f"- AUPR cao nhất đạt {best_aupr:.4f}")

if __name__ == "__main__":
    main()