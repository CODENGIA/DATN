Dưới đây là tài liệu báo cáo khối lượng công việc được viết lại toàn vẹn theo chuẩn học thuật, bổ sung chi tiết phần giải nghĩa cho từng ký hiệu toán học trong các công thức.

---

### 1. Chuẩn bị và biểu diễn đồ thị sinh học kép (Dual-Graph Representation)

Để mô phỏng sự thay đổi mạng lưới gen giữa trạng thái bình thường và ung thư, hệ thống xây dựng hai đồ thị song song chia sẻ cùng cấu trúc topo nhưng khác biệt về đặc trưng điểm nút (mức độ biểu hiện gen). Trọng số cạnh được tích hợp để biểu thị độ tin cậy của các tương tác protein-protein (PPI) thực tế.

Cho đồ thị $G = (V, E, W)$, mỗi nút $i$ đại diện cho một gen có vectơ đặc trưng ban đầu $x_i \in \mathbb{R}^{14}$. Các đặc trưng này được chuẩn hóa để định cỡ dữ liệu biểu hiện gen:

$$x_i^{(norm)} = \frac{x_i - \mu}{\sigma}$$

**Giải nghĩa ký hiệu:**

* $x_i^{(norm)}$: Vectơ đặc trưng của gen (nút) $i$ sau khi đã chuẩn hóa (Z-score).
* $x_i$: Vectơ đặc trưng thô ban đầu của gen $i$ (kích thước 14 chiều).
* $\mu$: Vectơ giá trị trung bình của toàn bộ các gen trong tập dữ liệu.
* $\sigma$: Vectơ độ lệch chuẩn của toàn bộ các gen trong tập dữ liệu.

---

### 2. Mạng nơ-ron đồ thị đệm trọng số cạnh (Edge-Weighted GATv2)

Trong mạng sinh học, không phải mọi tương tác protein đều có tầm quan trọng như nhau đối với quá trình phát sinh khối u. Thay vì coi mọi cạnh là bình đẳng, mô hình tích hợp $w_{ij}$ trực tiếp vào cơ chế Attention để điều hướng luồng thông tin, ưu tiên các tương tác có độ tin cậy sinh học cao và sử dụng kỹ thuật DropEdge để giảm nhiễu topo.

Cơ chế cập nhật nút ở lớp thứ $l$ được định nghĩa như sau:

$$e_{ij}^{(l)} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^{(l)} \parallel \mathbf{W} h_j^{(l)} \parallel w_{ij}]\right)$$

$$\alpha_{ij}^{(l)} = \frac{\exp(e_{ij}^{(l)})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^{(l)})}$$

$$h_i^{(l+1)} = \text{ELU}\left(h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} \mathbf{W} h_j^{(l)}\right)$$

**Giải nghĩa ký hiệu:**

* $e_{ij}^{(l)}$: Điểm chú ý (attention score) thô giữa gen $i$ và gen láng giềng $j$ tại lớp thứ $l$.
* $\mathbf{a}^T$: Vectơ trọng số chú ý (attention weight vector) có thể học được của mạng nơ-ron.
* $\mathbf{W}$: Ma trận trọng số phép biến đổi tuyến tính để chiếu đặc trưng gen.
* $h_i^{(l)}, h_j^{(l)}$: Vectơ biểu diễn ẩn (hidden state) của gen $i$ và gen $j$ tại lớp $l$.
* $\parallel$: Phép toán ghép nối các vectơ (Concatenation).
* $w_{ij}$: Trọng số cạnh vô hướng (scalar edge weight) thể hiện độ tin cậy của tương tác giữa gen $i$ và $j$.
* $\alpha_{ij}^{(l)}$: Hệ số chú ý đã được chuẩn hóa (có giá trị từ 0 đến 1) dùng để phân bổ tầm quan trọng cho gen $j$.
* $\mathcal{N}(i)$: Tập hợp tất cả các gen láng giềng có kết nối trực tiếp với gen $i$ trong đồ thị.
* $h_i^{(l+1)}$: Vectơ biểu diễn ẩn được cập nhật của gen $i$ cho lớp tiếp theo ($l+1$).
* $\text{LeakyReLU}, \text{ELU}$: Các hàm kích hoạt phi tuyến tính để giữ luồng đạo hàm ổn định.

---

### 3. Tổng hợp đa mức độ (Jumping Knowledge Network)

Các con đường truyền tín hiệu sinh học (signaling pathways) hoạt động ở nhiều cấp độ cấu trúc: từ các phức hợp protein cục bộ đến toàn bộ mạng lưới chuyển hóa. Cơ chế JK-Net gom các biểu diễn từ mọi độ sâu của lớp tích chập để chống lại hiện tượng "quá nhẵn" (over-smoothing), giữ lại cả đặc trưng sinh học cục bộ và toàn cục.

Đầu ra nhúng cuối cùng của một nhánh đồ thị được tính bằng:

$$Z_i = \mathbf{W}_{JK} \left[ h_i^{(1)} \parallel h_i^{(2)} \parallel h_i^{(3)} \right]$$

**Giải nghĩa ký hiệu:**

* $Z_i$: Vectơ biểu diễn đặc trưng đồ thị (graph embedding) tổng hợp cuối cùng của gen $i$ trên một nhánh.
* $\mathbf{W}_{JK}$: Ma trận trọng số chiếu (projection matrix) dùng để nén kích thước chiều của vectơ.
* $h_i^{(1)}, h_i^{(2)}, h_i^{(3)}$: Lần lượt là đầu ra biểu diễn ẩn của gen $i$ tại các lớp GAT thứ nhất, thứ hai và thứ ba.

---

### 4. Dung hợp chú ý chéo thưa thớt (Sparse Cross-Attention Fusion)

Để phát hiện gen đột biến điều khiển (driver genes), cần so sánh đối chiếu trạng thái của cùng một gen và môi trường lân cận của nó giữa mô bình thường và mô u. Việc triển khai cơ chế chú ý qua Message Passing giúp giới hạn không gian tính toán chỉ trên các tương tác có thực, giảm độ phức tạp từ $\mathcal{O}(N^2)$ xuống $\mathcal{O}(E)$ và loại bỏ nhiễu từ các gen không liên quan sinh lý học.

Quá trình dung hợp được biểu diễn toán học như sau:

$$Z_{attn, i} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \text{softmax}_j \left( \frac{(\mathbf{W}_Q Z_i^N) \cdot (\mathbf{W}_K Z_j^T)}{\sqrt{d}} \right) (\mathbf{W}_V Z_j^T)$$

$$Z_{diff, i} = (Z_i^N - Z_i^T)^2$$

$$Z_{fusion, i} = \mathbf{W}_f \left[ Z_{attn, i} \parallel Z_{diff, i} \right]$$

**Giải nghĩa ký hiệu:**

* $Z_{attn, i}$: Kết quả biểu diễn của gen $i$ sau khi lấy thông tin tương quan chéo.
* $\mathcal{N}(i) \cup \{i\}$: Tập hợp các gen láng giềng của gen $i$ cộng thêm chính nó (self-loop).
* $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$: Các ma trận trọng số chiếu lần lượt tạo ra Truy vấn (Query), Khóa (Key) và Giá trị (Value).
* $Z_i^N$: Biểu diễn gen $i$ ở trạng thái Bình thường (Normal) – đóng vai trò làm Query.
* $Z_j^T$: Biểu diễn gen $j$ ở trạng thái Khối u (Tumor) – đóng vai trò làm Key và Value.
* $\cdot$: Phép tính tích vô hướng (Dot-product).
* $d$: Kích thước không gian ẩn của một head chú ý (được dùng để chuẩn hóa tỷ lệ).
* $Z_{diff, i}$: Vectơ chứa phần dư trực tiếp, tính bằng bình phương độ lệch giữa trạng thái Bình thường và Khối u của gen $i$.
* $Z_{fusion, i}$: Vectơ đặc trưng dung hợp cuối cùng, sẵn sàng để phân loại.
* $\mathbf{W}_f$: Ma trận trọng số tổng hợp kết quả của Cross-Attention và Residual Hint.

---

### 5. Ước lượng tỷ lệ tiên nghiệm (Dynamic Prior Estimation via AlphaMax)

Tỷ lệ gen điều khiển khối u thực tế khác nhau tùy thuộc vào từng loại bệnh lý (ví dụ: gánh nặng đột biến ở ung thư da khác với ung thư tuyến giáp). Mô hình áp dụng kỹ thuật tiệm cận AlphaMax để ước lượng phân phối tiên nghiệm $\pi$ trực tiếp từ dữ liệu thay vì sử dụng một hằng số cố định, phản ánh đúng đặc tính dịch tễ của từng tập dữ liệu.

Công thức tính tỷ lệ tiên nghiệm được giới hạn như sau:

$$\pi = \text{clip}\left( \frac{\mathbb{E}_{x \in P}[\hat{P}(Y=1|x)]}{\max(\mathbb{E}_{x \in U}[\hat{P}(Y=1|x)], \epsilon)}, \text{low}, \text{high} \right)$$

**Giải nghĩa ký hiệu:**

* $\pi$: Xác suất tiên nghiệm (prior probability) ước lượng tỷ lệ gen ung thư trong tập dữ liệu.
* $\text{clip}$: Hàm giới hạn giá trị trong một khoảng cho trước (từ `low` đến `high`).
* $P, U$: Tập hợp các gen Đã biết là Ung thư (Positive) và tập các gen Chưa gán nhãn (Unlabeled).
* $\mathbb{E}_{x \in P}, \mathbb{E}_{x \in U}$: Giá trị kỳ vọng (trung bình toán học) đối với các mẫu thuộc tập $P$ và tập $U$.
* $\hat{P}(Y=1|x)$: Xác suất dự đoán mẫu $x$ là nhãn dương, xuất ra từ một mô hình hồi quy logistic độc lập sơ bộ.
* $\epsilon$: Hằng số bảo vệ (ví dụ: $1e-8$) để tránh lỗi chia cho $0$.
* $\text{low}, \text{high}$: Các ngưỡng biên giới hạn an toàn để tỷ lệ tiên nghiệm không quá nhỏ hoặc quá lớn (thường đặt ở $0.01$ và $0.15$).

---

### 6. Hàm rủi ro học máy PU không âm (Robust Non-negative PU Risk)

Trong nghiên cứu ung thư, các gen chưa được phát hiện không đồng nghĩa với việc chúng vô hại (không thể gán nhãn âm hoàn toàn). Hàm mất mát nnPU xử lý việc khuyết nhãn này bằng cách ước lượng và tối thiểu hóa rủi ro thực nghiệm. Cơ chế "Gradient Flip" được thiết lập nhằm ngăn chặn sự sụp đổ gradient (hội tụ giả) khi đánh giá dữ liệu sinh học mất cân bằng nghiêm trọng.

Hàm rủi ro được định nghĩa ban đầu là:

$$\widehat{R}_{PU} = \pi \widehat{R}_P^+ + \max\left(0, \widehat{R}_U^- - \pi \widehat{R}_P^-\right)$$

Trong trường hợp rủi ro cấu thành $R_{neg} = \widehat{R}_U^- - \pi \widehat{R}_P^- < 0$, thay vì để gradient triệt tiêu, hàm mục tiêu được lật dấu để duy trì luồng học:

$$\mathcal{L} = \pi \widehat{R}_P^+ - R_{neg}$$

**Giải nghĩa ký hiệu:**

* $\widehat{R}_{PU}$: Hàm ước lượng rủi ro tổng thể của bài toán Positive-Unlabeled Learning.
* $\pi$: Tỷ lệ tiên nghiệm (đã tính ở phần AlphaMax).
* $\widehat{R}_P^+$: Rủi ro thực nghiệm trên tập Positive (mức độ mô hình phân loại sai gen ung thư đã biết thành gen không gây bệnh).
* $\widehat{R}_U^-$: Rủi ro thực nghiệm trên tập Unlabeled, được tính bằng cách giả định (tạm thời) toàn bộ tập Unlabeled đều là nhãn Âm (Negative).
* $\widehat{R}_P^-$: Rủi ro phân loại sai các mẫu Positive thành nhãn Âm. Thành phần $\pi \widehat{R}_P^-$ đóng vai trò là "chất bù trừ" để điều chỉnh lại lượng rủi ro bị tính lố trong $\widehat{R}_U^-$.
* $R_{neg}$: Phần rủi ro ước lượng đại diện cho lớp Negative thực sự ($R_{neg} = \widehat{R}_U^- - \pi \widehat{R}_P^-$).
* $\mathcal{L}$: Hàm mục tiêu cuối cùng sau khi xử lý lật dấu (Gradient Flip) để huấn luyện mô hình.
