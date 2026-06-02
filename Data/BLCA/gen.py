import pandas as pd

def filter_and_match_genes(target_num_genes=6500):
    # 1. Đọc dữ liệu từ các file TSV
    print("Đang đọc dữ liệu từ các file...")
    df_candidate = pd.read_csv('1_BLCA_candidate.tsv', sep='\t')
    df_ncg = pd.read_csv('BLCA_pos.tsv', sep='\t')
    df_oncokb = pd.read_csv('BLCA_oncokb_biomarker_drug_associations.tsv', sep='\t')
    
    # Đọc file index (do file này cột đầu tiên không có tên header nên ta đặt tên cho nó)
    df_index = pd.read_csv('BLCA_gene_index_humannet.tsv', sep='\t', names=['gene', 'index'], header=0)


    # 2. Trích xuất danh sách gen gây ung thư
    ncg_genes = set(df_ncg['symbol'].dropna())
    oncokb_genes = set(df_oncokb['Gene'].dropna())
    all_cancer_genes = ncg_genes.union(oncokb_genes)

    # Lọc ra những gen ung thư thực sự tồn tại trong danh sách candidate
    candidate_genes = set(df_candidate['gene'].dropna())
    target_genes_in_candidate = all_cancer_genes.intersection(candidate_genes)
    
    # 3. Tìm vị trí sâu nhất chứa gen ung thư
    max_index = 0
    for idx, row in df_candidate.iterrows():
        if row['gene'] in target_genes_in_candidate:
            max_index = idx
            
    min_required_genes = max_index + 1
    print(f"Số lượng gen tối thiểu cần thiết để bao trọn gen gây ung thư là: {min_required_genes}")

    # 4. Hiệu chỉnh số lượng gen xuất ra
    # Nếu bạn muốn 5000 gen nhưng chỉ cần 3500 gen là đủ chứa gen ung thư -> lấy 5000
    # Nếu bạn muốn 3000 gen nhưng cần 3500 gen mới đủ chứa gen ung thư -> bắt buộc lấy 3500
    final_num_genes = max(target_num_genes, min_required_genes)
    if final_num_genes > target_num_genes:
        print(f"CẢNH BÁO: Số lượng bạn yêu cầu ({target_num_genes}) quá nhỏ để bao hàm đủ gen ung thư. Tự động điều chỉnh lên {final_num_genes} gen.")
    else:
        print(f"Đang lấy {final_num_genes} gen đứng đầu từ danh sách candidate...")

    df_filtered = df_candidate.iloc[:final_num_genes]

    # 5. Đối chiếu với file LUAD_gene_index_humannet.tsv
    print("Đang đối chiếu với file index để lấy gen huấn luyện mô hình...")
    # Lấy các gen vừa lọc có tồn tại trong file index (Inner join)
    df_final_training = pd.merge(df_filtered, df_index, on='gene', how='inner')
    
    # 6. Xuất ra file
    output_filename = f'training_genes_{final_num_genes}.tsv'
    df_final_training.to_csv(output_filename, sep='\t', index=False)

    print(f"--- HOÀN TẤT ---")
    print(f"Tổng số gen có trong file kết quả (sau khi đối chiếu file index): {len(df_final_training)}")
    print(f"Đã lưu kết quả sẵn sàng để huấn luyện vào file: {output_filename}")

if __name__ == "__main__":
    # BẠN CÓ THỂ THAY ĐỔI SỐ LƯỢNG GEN MONG MUỐN Ở ĐÂY
    # Ví dụ: sửa thành 5000 nếu muốn xuất 5000 gen
    SO_LUONG_GEN_MONG_MUON = 6500
    filter_and_match_genes(target_num_genes=SO_LUONG_GEN_MONG_MUON)