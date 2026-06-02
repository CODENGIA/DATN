import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import gc
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from torch_geometric.nn import GATv2Conv

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

# ==============================================================================
# KIẾN TRÚC MÔ HÌNH MẠNG NƠ-RON ĐỒ THỊ
# ==============================================================================
class ModernPUGNN(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        classifier_in_dim = hidden_dim * 2 + in_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def get_embedding(self, x, edge_index):
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.elu(self.conv1(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.elu(self.conv2(z, edge_index))
        return z

    def forward(self, x_n, x_t, edge_index_n, edge_index_t):
        z_n = self.get_embedding(x_n, edge_index_n)
        z_t = self.get_embedding(x_t, edge_index_t)
        
        raw_diff = torch.abs(x_n - x_t)
        z_final = torch.cat([z_n, z_t, raw_diff], dim=-1)
        
        logits = self.classifier(z_final).squeeze(-1)
        return logits

# ==============================================================================
# HÀM MẤT MÁT NON-NEGATIVE PU LOSS
# ==============================================================================
def nnpu_loss(logits, labels, prior=0.05):
    pos_mask = (labels == 1.0).float()
    unl_mask = (labels == 0.0).float()
    
    num_pos = max(pos_mask.sum().item(), 1.0)
    num_unl = max(unl_mask.sum().item(), 1.0)
    
    loss_pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    term_pos = (loss_pos * pos_mask).sum() / num_pos
    
    loss_unl_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    term_unl_neg = (loss_unl_neg * unl_mask).sum() / num_unl
    
    loss_pos_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    term_pos_neg = (loss_pos_neg * pos_mask).sum() / num_pos
    
    risk_neg = term_unl_neg - prior * term_pos_neg
    
    if risk_neg < 0:
        objective = prior * term_pos - risk_neg
    else:
        objective = prior * term_pos + risk_neg
        
    return objective

from utils import prepare_dual_graph_data, load_and_split_ground_truth

def get_all_cancer_folders(base_data_dir):
    ignore_list = ['CPDB', 'STRING']
    if not os.path.exists(base_data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục gốc: {base_data_dir}")
        
    all_folders = os.listdir(base_data_dir)
    cancer_types = [f for f in all_folders if os.path.isdir(os.path.join(base_data_dir, f)) and f not in ignore_list]
    return sorted(cancer_types)

# ==============================================================================
# PIPELINE FINE-TUNE VÀ ĐÁNH GIÁ TỰ ĐỘNG (GIAI ĐOẠN 2)
# ==============================================================================
def finetune_and_evaluate_pipeline(base_data_dir, save_dir, target_cancers, target_num_genes=9000, epochs=200, seed=42):
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_path = os.path.join(save_dir, 'pretrained_base_model.pth')
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"LỖI: Không tìm thấy file {pretrained_path}. Phải chạy stage1_pretrain.py trước.")
        
    print(f"[*] Chạy Pipeline trên thiết bị: {device}")
    print(f"[*] Các bệnh sẽ được rẽ nhánh Fine-tune: {target_cancers}\n")
    
    for cancer in target_cancers:
        print("=" * 80)
        print(f"🚀 BẮT ĐẦU FINE-TUNE & CHẤM ĐIỂM CHUẨN KHOA HỌC: {cancer}")
        print("=" * 80)
        
        # 1. TẢI LẠI TRỌNG SỐ BASE GỐC CHO MỖI BỆNH (Tránh lai tạp)
        model = ModernPUGNN(in_dim=14, hidden_dim=64).to(device)
        # Bỏ strict=False nếu muốn kiểm tra nghiêm ngặt trọng số
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True), strict=False)
        
        scaler = torch.amp.GradScaler('cuda')

        # 2. NẠP VÀ CHIA DỮ LIỆU ĐỒ THỊ ĐẶC THÙ
        cancer_dir = os.path.join(base_data_dir, cancer)
        pkl_path = os.path.join(cancer_dir, f'{cancer}_input_data_humannet.pkl')
        orig_tsv = os.path.join(cancer_dir, f'{cancer}_gene_index_humannet.tsv')
        target_tsv = os.path.join(cancer_dir, f'{cancer}_training_genes_{target_num_genes}.tsv')
        ncg_path = os.path.join(cancer_dir, f'{cancer}_pos.tsv')
        oncokb_path = os.path.join(cancer_dir, f'{cancer}_oncokb_biomarker_drug_associations.tsv')
        
        x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(pkl_path, orig_tsv, target_tsv, device)
        Y_train, Y_full, eval_mask = load_and_split_ground_truth(ncg_path, oncokb_path, current_gene_list, train_ratio=0.7, seed=42, device=device)

        num_pos_train = (Y_train == 1.0).sum().item()
        total_genes = len(Y_train)
        calculated_prior = max(num_pos_train / total_genes, 0.001)
        
        # 3. CHIẾN LƯỢC QUẢN LÝ TRỌNG SỐ ĐỘNG (DỰA TRÊN ĐỘ PHỨC TẠP CỦA TẬP NHÃN)
        if num_pos_train > 30:
            # Bệnh giàu dữ liệu (>30 gen Train): Đóng băng conv1, mở khóa conv2 với lr=1e-5 để tinh chỉnh topology
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = True
                
            optimizer = torch.optim.AdamW([
                {'params': model.conv2.parameters(), 'lr': 1e-5},
                {'params': model.classifier.parameters(), 'lr': 1e-4}
            ], weight_decay=1e-4)
            print(f"    [-->] ĐĂNG KÝ: Bệnh dồi dào nhãn ({int(num_pos_train)} gen Train). Mở khóa lớp conv2 với lr=1e-5.")
        else:
            # Bệnh hiếm (<=30 gen Train): Đóng băng 100% mạng GNN để triệt tiêu hiện tượng overfit dữ liệu ít
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
                
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
            print(f"    [-->] ĐĂNG KÝ: Bệnh thưa thớt nhãn ({int(num_pos_train)} gen Train). Khóa cứng GNN, chỉ huấn luyện Classifier.")

        print(f"[*] Tham số Prior nnPU được thiết lập: {calculated_prior:.6f}")
        
        # 4. TIẾN HÀNH FINE-TUNING LỚP PHÂN LOẠI / ĐỒ THỊ CUỐI
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Lưu ý: truyền thêm e_t vào hàm forward
                logits = model(x_n, x_t, e_n, e_t)
                loss = nnpu_loss(logits, Y_train, prior=calculated_prior)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if (epoch + 1) == epochs or (epoch + 1) % 50 == 0:
                print(f"    - Epoch {epoch+1:03d}/{epochs} | nnPU Loss: {loss.item():.4f}")

        # Lưu lại mô hình chuyên biệt cho căn bệnh này
        specific_save_path = os.path.join(save_dir, f'finetuned_model_{cancer}.pth')
        torch.save(model.state_dict(), specific_save_path)

        # 5. ĐÁNH GIÁ HIỆU NĂNG TRÊN TẬP ĐỘC LẬP (INDEPENDENT SET - KHÔNG DATA LEAKAGE)
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP ĐỘC LẬP (INDEPENDENT SET):")
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(x_n, x_t, e_n, e_t)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Khử toàn bộ các phần tử đã được xem nhãn trong tập Train
            mask_np = eval_mask.cpu().numpy()
            y_unseen = Y_full.cpu().numpy()[mask_np]
            probs_unseen = probs[mask_np]
            
            # Thiết lập tập Validation độc lập để dò tìm ngưỡng và tập Test để ghi nhận điểm
            indices = np.arange(len(y_unseen))
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
            
            split_point = int(len(y_unseen) * 0.5)
            val_idx = indices[:split_point]
            test_idx = indices[split_point:]
            
            y_val, probs_val = y_unseen[val_idx], probs_unseen[val_idx]
            y_test, probs_test = y_unseen[test_idx], probs_unseen[test_idx]
            
            # TÌM NGƯỠNG TỐI ƯU TRÊN TẬP VALIDATION
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
            denominator = (precisions + recalls)
            denominator[denominator == 0] = 1e-8
            f1_scores_val = 2 * (precisions * recalls) / denominator
            
            best_idx = np.argmax(f1_scores_val)
            VALIDATION_THRESH = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            
            # KIỂM THỬ THẬT SỰ TRÊN TẬP TEST MÙ
            auc_score = roc_auc_score(y_test, probs_test)
            aupr_score = average_precision_score(y_test, probs_test)
            
            preds_test = (probs_test > VALIDATION_THRESH).astype(float)
            final_f1 = f1_score(y_test, preds_test)
            
            print(f"    - ROC-AUC         : {auc_score:.4f}")
            print(f"    - AUPRC (Đề xuất) : {aupr_score:.4f} <-- Chỉ số đối chiếu chính")
            print(f"    - Ngưỡng từ Val   : {VALIDATION_THRESH:.6f}")
            print(f"    - F1-Score (Test) : {final_f1:.4f}")
            print(f"    - Dự đoán Positive: {int(preds_test.sum())} gen\n")

        # 6. QUÉT SẠCH VRAM SAU MỖI VÒNG LẶP ĐỂ CHỐNG LỖI OUT-OF-MEMORY
        del model, optimizer, logits, probs, x_n, e_n, x_t, e_t, Y_full, Y_train, eval_mask
        del y_unseen, probs_unseen, y_val, probs_val, y_test, probs_test, indices, preds_test
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    BASE_DIR = '/content/drive/MyDrive/DATN/Data'
    SAVE_DIR = '/content/drive/MyDrive/DATN/Checkpoints'
    
    all_cancers = get_all_cancer_folders(BASE_DIR)
    
    # Loại trừ các bệnh nền tảng đã dốc toàn lực học ở Giai đoạn 1
    pretrain_cancers = ['BRCA', 'LUAD'] 
    cancers_to_finetune = [c for c in all_cancers if c not in pretrain_cancers]
    
    # Kích hoạt toàn bộ chu trình xử lý tự động
    finetune_and_evaluate_pipeline(
        base_data_dir=BASE_DIR, 
        save_dir=SAVE_DIR, 
        target_cancers=cancers_to_finetune, 
        target_num_genes=9000,
        epochs=200, 
        seed=42
    )