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
# HÀM BỔ TRỢ & MÔ HÌNH (Gộp trực tiếp vào để bạn không phải lo import thiếu)
# ==============================================================================
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

class ModernPUGNN(nn.Module):
    """
    Kiến trúc Early Fusion GATv2 kết hợp Residual Skip-Connection.
    Ép mô hình chú ý vào độ lệch sinh học thuần túy.
    """
    def __init__(self, in_dim=14, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        combined_dim = in_dim * 2
        
        self.conv1 = GATv2Conv(combined_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        classifier_in_dim = hidden_dim + in_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_n, x_t, edge_index):
        x_combined = torch.cat([x_n, x_t], dim=-1)
        z = F.dropout(x_combined, p=self.dropout, training=self.training)
        z = F.elu(self.conv1(z, edge_index))
        
        z = F.dropout(z, p=self.dropout, training=self.training)
        z_gnn = F.elu(self.conv2(z, edge_index))
        
        raw_diff = torch.abs(x_n - x_t)
        
        z_final = torch.cat([z_gnn, raw_diff], dim=-1)
        
        logits = self.classifier(z_final).squeeze(-1)
        return logits

def nnpu_loss(logits, labels, prior=0.05):
    """
    Non-negative PU Loss kinh điển.
    """
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

# Hãy đảm bảo utils.py đã được cập nhật hàm load_and_split_ground_truth
from utils import prepare_dual_graph_data, load_and_split_ground_truth

# ==============================================================================
# QUÁ TRÌNH HUẤN LUYỆN PAN-CANCER (GIAI ĐOẠN 1)
# ==============================================================================
def main_pretrain():
    seed_everything(42)

    parser = argparse.ArgumentParser(description="Giai đoạn 1: Pre-training GATv2 trên các bệnh lớn")
    parser.add_argument('--base_data_dir', type=str, default='/content/drive/MyDrive/DATN/Data')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0005) 
    
    args, unknown = parser.parse_known_args()

    save_dir = '/content/drive/MyDrive/DATN/Checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Thiết bị Pre-training: {device} ---")

    # Khởi tạo MỚI HOÀN TOÀN
    model = ModernPUGNN(in_dim=14, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    early_stopping = EarlyStopping(patience=40, min_delta=1e-5)
    
    # DANH SÁCH CÁC BỆNH LỚN DÙNG ĐỂ MÀI GIŨA (Tùy chọn)
    rich_cancers = ['LUAD', 'BRCA'] 
    
    print("\n================ BẮT ĐẦU GIAI ĐOẠN 1: PRE-TRAINING PAN-CANCER ===================")
    
    for cancer_type in rich_cancers:
        print(f"\n[*] Đang nạp tri thức từ bệnh: {cancer_type}")
        
        cancer_dir = os.path.join(args.base_data_dir, cancer_type)
        pkl_path = os.path.join(cancer_dir, f'{cancer_type}_input_data_humannet.pkl')        
        orig_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_gene_index_humannet.tsv')   
        target_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_training_genes_9000.tsv')      
        ncg_path = os.path.join(cancer_dir, f'{cancer_type}_pos.tsv')
        oncokb_path = os.path.join(cancer_dir, f'{cancer_type}_oncokb_biomarker_drug_associations.tsv')
        
        # 2. LOAD VÀ CHIA DỮ LIỆU
        x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
            pkl_path, orig_tsv_path, target_tsv_path, device
        )

        Y_train, Y_full, eval_mask = load_and_split_ground_truth(
            ncg_path, oncokb_path, current_gene_list, train_ratio=0.7, seed=42, device=device
        )
        
        # Tính prior cho tập Train hiện tại
        num_pos_train = (Y_train == 1.0).sum().item()
        calculated_prior = max(num_pos_train / len(Y_train), 0.01)

        # 3. HUẤN LUYỆN
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(x_n, x_t, e_n)
                # Chỉ học trên những nhãn Train đã chia
                loss = nnpu_loss(logits, Y_train, prior=calculated_prior)

            scaler.scale(loss).backward()
            scaler.step(optimizer)  
            scaler.update()
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Bệnh {cancer_type} | Epoch {epoch+1:03d}/{args.epochs} | nnPU Loss: {loss.item():.4f}")

            # Lưu ý: Early stopping có thể reset lại tùy bạn khi sang bệnh mới
            early_stopping(loss.item(), model)
            if early_stopping.early_stop:
                print(f"--> [KÍCH HOẠT] Dừng sớm tại Epoch {epoch+1}")
                early_stopping.early_stop = False # Reset cho bệnh tiếp theo
                early_stopping.counter = 0
                break

        if early_stopping.best_weights is not None:
            model.load_state_dict({k: v.to(device) for k, v in early_stopping.best_weights.items()})

    # Lưu lại model gốc làm điểm xuất phát cho Stage 2
    model_filename = f'pretrained_base_model.pth'
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\n[+] Đã lưu TRỌNG SỐ GỐC vào: {model_path}")
    
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main_pretrain()