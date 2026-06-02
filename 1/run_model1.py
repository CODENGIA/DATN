import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc

warnings.filterwarnings("ignore", category=FutureWarning)

from utils import prepare_dual_graph_data, load_ground_truth_labels
from model import DualGraphCrossAttentionModel, info_nce_loss

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Dual-Graph Cross-Attention with InfoNCE")
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/DATN/Data/BLCA', help='Thư mục chứa dữ liệu')
    parser.add_argument('--epochs', type=int, default=100, help='Số vòng lặp huấn luyện')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang chạy mô hình trên: {device} ---")

    # 1. KHAI BÁO ĐƯỜNG DẪN 3 FILE DỮ LIỆU
    pkl_path = os.path.join(args.data_dir, 'BLCA_input_data_humannet.pkl')        
    orig_tsv_path = os.path.join(args.data_dir, 'BLCA_gene_index_humannet.tsv')   
    target_tsv_path = os.path.join(args.data_dir, 'training_genes_6500.tsv')      
    
    # Hàm lọc động
    x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv_path, device
    )

    # 2. NẠP ĐÁP ÁN CHUẨN (Ground Truth)
    ncg_path = os.path.join(args.data_dir, 'BLCA_pos.tsv')
    oncokb_path = os.path.join(args.data_dir, 'BLCA_oncokb_biomarker_drug_associations.tsv')
    Y_test = load_ground_truth_labels(ncg_path, oncokb_path, current_gene_list, device)

    # 3. KHỞI TẠO MÔ HÌNH
    model = DualGraphCrossAttentionModel(input_dim=14, embed_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    print("\n================ BẮT ĐẦU HUẤN LUYỆN TỰ GIÁM SÁT ================")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Gọi 2 biến out_n, out_t (Vì cấu trúc model.py hiện tại chỉ trả về 2 biến)
            out_n, out_t = model(x_n, e_n, x_t, e_t)
            
            # Sử dụng InfoNCE: temperature 0.1 và 5 negative samples
            loss = info_nce_loss(out_n, out_t, temperature=0.05, num_negatives=5, return_individual_loss=False)

        scaler.scale(loss).backward()
        scaler.step(optimizer)  
        scaler.update()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {loss.item():.4f}")

    torch.cuda.empty_cache()
    gc.collect()

    print("\n================ ĐÁNH GIÁ BẰNG 'ĐỘ KHÁNG CỰ INFO-NCE' ================")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out_n, out_t = model(x_n, e_n, x_t, e_t)
            
            # Trích xuất điểm loss cá nhân của từng gen (Tăng num_negatives lên 20 để test ranh giới gắt hơn)
            individual_losses = info_nce_loss(out_n, out_t, temperature=0.05, num_negatives=20, return_individual_loss=True)
        
        # Loss của gen nào càng cao -> Gen đó kháng cự lại việc uốn nắn càng mạnh -> Bất thường
        anomaly_scores = individual_losses
        
        # Chuẩn hóa Z-score
        mean_s = anomaly_scores.mean()
        std_s = anomaly_scores.std()
        anomaly_scores_norm = (anomaly_scores - mean_s) / (std_s + 1e-8)
        
        probs_np = torch.sigmoid(anomaly_scores_norm).cpu().numpy()
        y_test_np = Y_test.cpu().numpy()
        
        precisions, recalls, thresholds = precision_recall_curve(y_test_np, probs_np)
        
        # Bổ sung tính toán AUPRC bằng hàm auc
        auprc = auc(recalls, precisions)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        print(f"[*] Ngưỡng tự chọn để tối ưu F1: {best_threshold:.4f}")
        preds_np = (probs_np > best_threshold).astype(float) 
        
        auc_score = roc_auc_score(y_test_np, probs_np)
        aupr = average_precision_score(y_test_np, probs_np)
        f1 = f1_score(y_test_np, preds_np)
        
        print(f">>> KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ <<<")
        print(f"AUC    = {auc_score:.4f}")
        print(f"AUPR   = {aupr:.4f}")
        print(f"AUPRC  = {auprc:.4f}")
        print(f"F1     = {f1:.4f}")

if __name__ == "__main__":
    main()