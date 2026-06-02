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
# HÀM BỔ TRỢ & MÔ HÌNH (Đã loại bỏ nnPU loss vì giấu nhãn)
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
    Kiến trúc Late Fusion GATv2 kết hợp Residual Skip-Connection.
    """
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

def info_nce_loss(z1, z2, temperature=0.1):
    """
    Hàm mất mát Contrastive Learning (InfoNCE).
    Ép vector của cùng một gen ở mạng Normal và Tumor lại gần nhau.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Tính ma trận tương đồng Cosine
    logits = torch.matmul(z1, z2.T) / temperature
    
    # Nhãn là đường chéo (gen i ở đồ thị 1 phải khớp gen i ở đồ thị 2)
    labels = torch.arange(z1.size(0)).to(z1.device)
    
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)
    
    return (loss_1 + loss_2) / 2

# Đảm bảo utils.py chứa prepare_dual_graph_data
from utils import prepare_dual_graph_data

# ==============================================================================
# QUÁ TRÌNH HUẤN LUYỆN CONTRASTIVE PAN-CANCER (GIAI ĐOẠN 1)
# ==============================================================================
def main_pretrain():
    seed_everything(42)

    parser = argparse.ArgumentParser(description="Giai đoạn 1: Contrastive Pre-training GATv2 (Hoàn toàn giấu nhãn)")
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
    
    print("\n================ BẮT ĐẦU GIAI ĐOẠN 1: CONTRASTIVE PRE-TRAINING ===================")
    
    for cancer_type in rich_cancers:
        print(f"\n[*] Đang nạp tri thức mạng lưới từ bệnh: {cancer_type} (KHÔNG DÙNG NHÃN)")
        
        cancer_dir = os.path.join(args.base_data_dir, cancer_type)
        pkl_path = os.path.join(cancer_dir, f'{cancer_type}_input_data_humannet.pkl')        
        orig_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_gene_index_humannet.tsv')   
        target_tsv_path = os.path.join(cancer_dir, f'{cancer_type}_training_genes_9000.tsv')      
        
        # 2. CHỈ LOAD ĐỒ THỊ VÀ FEATURES, KHÔNG LOAD NHÃN GROUND TRUTH
        x_n, e_n, x_t, e_t, current_gene_list = prepare_dual_graph_data(
            pkl_path, orig_tsv_path, target_tsv_path, device
        )

        # 3. HUẤN LUYỆN CONTRASTIVE
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Trích xuất vector biểu diễn (embeddings) từ 2 đồ thị
                z_n = model.get_embedding(x_n, e_n)
                z_t = model.get_embedding(x_t, e_t)
                
                # Hàm mất mát InfoNCE
                loss = info_nce_loss(z_n, z_t, temperature=0.1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)  
            scaler.update()
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Bệnh {cancer_type} | Epoch {epoch+1:03d}/{args.epochs} | InfoNCE Loss: {loss.item():.4f}")

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
    print(f"\n[+] Đã lưu TRỌNG SỐ TỔNG QUÁT GỐC vào: {model_path}")
    
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main_pretrain()