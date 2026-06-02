import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import random
import argparse
import gc
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

warnings.filterwarnings("ignore", category=FutureWarning)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# [CẬP NHẬT] Import hàm load_and_split_ground_truth từ utils mới
from utils import prepare_dual_graph_data, load_and_split_ground_truth
from modelVIP import ModernPUGNN, nnpu_loss

class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-5): 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None

    def __call__(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def main(cancer_type, target_num_genes):
    seed_everything(42)

    parser = argparse.ArgumentParser(description="Mô hình SOTA: GATv2 + Residual + nnPU Loss (Transductive)")
    parser.add_argument('--base_data_dir', type=str, default='/content/drive/MyDrive/DATN/Data')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0005) 
    
    args, unknown = parser.parse_known_args()

    save_dir = '/content/drive/MyDrive/DATN/Checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang chạy mô hình GATv2 cho bệnh {cancer_type} trên: {device} ---")

    # 1. THIẾT LẬP ĐƯỜNG DẪN ĐỘNG THEO TÊN BỆNH
    cancer_dir = os.path.join(args.base_data_dir, cancer_type)
    
    pkl_path = os.path.join(cancer_dir, f'{cancer_type}_input_data_humannet.pkl')        
    orig_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_gene_index_humannet.tsv')   
    target_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_training_genes_{target_num_genes}.tsv')      
    ncg_path = os.path.join(cancer_dir, f'{cancer_type}_pos.tsv')
    oncokb_path = os.path.join(cancer_dir, f'{cancer_type}_oncokb_biomarker_drug_associations.tsv')
    
    # 2. LOAD DỮ LIỆU ĐỒ THỊ
    x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
        pkl_path, orig_tsv_path, target_tsv_path, device
    )

    # [CẬP NHẬT] Sử dụng hàm chia nhãn an toàn từ utils.py (Chia 80% Train / 20% Test như code gốc của bạn)
    Y_train, Y_full, eval_mask = load_and_split_ground_truth(
        ncg_path, oncokb_path, current_gene_list, train_ratio=0.8, seed=42, device=device
    )

    # Tự động tính Prior thay vì dùng args.prior cố định (Phù hợp hơn với dữ liệu thực tế)
    num_pos_train = (Y_train == 1.0).sum().item()
    calculated_prior = max(num_pos_train / len(Y_train), 0.001)
    print(f"[*] Prior động cho bệnh {cancer_type}: {calculated_prior:.6f}")

    # 3. KHỞI TẠO MODEL
    model = ModernPUGNN(in_dim=14, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    early_stopping = EarlyStopping(patience=40, min_delta=1e-5)

    print("\n================ BẮT ĐẦU HUẤN LUYỆN NNPU ===================")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(x_n, x_t, e_n)
            # Học dựa trên Y_train (đã giấu bớt gen bệnh)
            loss = nnpu_loss(logits, Y_train, prior=calculated_prior)

        scaler.scale(loss).backward()
        scaler.step(optimizer)  
        scaler.update()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | nnPU Loss: {loss.item():.4f}")

        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            print(f"--> [KÍCH HOẠT] Dừng sớm tại Epoch {epoch+1}")
            break

    if early_stopping.best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in early_stopping.best_weights.items()})

    # Lưu file model
    model_filename = f'gatv2_residual_nnpu_transductive_{cancer_type}.pth'
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\n[+] Đã lưu mô hình tốt nhất vào: {model_path}")
    
    torch.cuda.empty_cache()
    gc.collect()

    # 4. ĐÁNH GIÁ MÔ HÌNH (CHUẨN SOTA: CHỐNG DATA LEAKAGE)
    print("\n================ KẾT QUẢ PHÂN LOẠI (INDEPENDENT SET) ===================")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(x_n, x_t, e_n)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # Lọc lấy những gen thuộc tập ẩn (eval_mask)
        mask_np = eval_mask.cpu().numpy()
        y_unseen = Y_full.cpu().numpy()[mask_np]
        probs_unseen = probs[mask_np]
        
        # [CẬP NHẬT TRỌNG TÂM]: Chia Validation và Test để không rò rỉ dữ liệu khi tìm Threshold
        indices = np.arange(len(y_unseen))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        
        split_point = int(len(y_unseen) * 0.5)
        val_idx = indices[:split_point]
        test_idx = indices[split_point:]
        
        y_val, probs_val = y_unseen[val_idx], probs_unseen[val_idx]
        y_test, probs_test = y_unseen[test_idx], probs_unseen[test_idx]
        
        # 4.1. Tìm Threshold tối ưu trên tập Validation
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
        denominator = (precisions + recalls)
        denominator[denominator == 0] = 1e-8
        f1_scores_val = 2 * (precisions * recalls) / denominator
        
        best_idx = np.argmax(f1_scores_val)
        VALIDATION_THRESH = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        # 4.2. Áp dụng Threshold đó để thi trên tập Test
        auc_score = roc_auc_score(y_test, probs_test)
        aupr = average_precision_score(y_test, probs_test)
        
        preds_test = (probs_test > VALIDATION_THRESH).astype(float) 
        final_f1 = f1_score(y_test, preds_test)
        
        print(f"[*] THỐNG KÊ HIỆU SUẤT TRÊN TẬP ẨN CỦA {cancer_type}:")
        print(f"- Ngưỡng xác suất tối ưu (Học từ Val): {VALIDATION_THRESH:.6f}")
        print(f"- ROC-AUC:  {auc_score:.4f}")
        print(f"- AUPRC:    {aupr:.4f} (Quan trọng nhất)")
        print(f"- F1-Score: {final_f1:.4f}")
        print(f"- Dự đoán Positive (Test): {int(preds_test.sum())} gen")

if __name__ == "__main__":
    # CHỈ CẦN THAY ĐỔI 2 BIẾN DƯỚI ĐÂY KHI CHUYỂN SANG BỆNH KHÁC
    CANCER_TYPE = 'KIRP' # Ví dụ: đổi thành 'LUAD', 'KIRC', 'BLCA'
    NUM_GENES = 9000     # Số lượng gen huấn luyện tương ứng
    
    main(cancer_type=CANCER_TYPE, target_num_genes=NUM_GENES)