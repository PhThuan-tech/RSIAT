# Ghi chú hướng nghiên cứu UMT-RSIAT

## 1. Phạm vi và trạng thái của tài liệu

Tài liệu này trình bày cơ sở hình thành, mục tiêu và công thức của biến thể đang
được phát triển trong repository, tạm gọi là **UMT-RSIAT** (*Weakly Nonlinear,
Moment-Aware Transport RSIAT with Adaptive Top-K Separation*).

Phạm vi hiện tại là class-incremental learning trên ảnh với các đặc điểm:

- backbone ViT-B/16 tiền huấn luyện;
- một shared adapter được cập nhật tuần tự;
- không lưu ảnh mẫu của các lớp cũ để huấn luyện lại;
- task boundary và tập lớp của mỗi task được biết khi huấn luyện;
- mô hình suy luận không chứa projector và không cần task ID.

Các công thức và mô tả dưới đây phản ánh code đã được triển khai. Đây là
**thiết kế và giả thuyết nghiên cứu**, chưa phải kết luận rằng UMT-RSIAT tốt hơn
RSIAT. Kết luận đó chỉ được đưa ra sau khi full run hoàn tất và có so sánh với
baseline trên cùng dữ liệu, class order, seed và ngân sách huấn luyện.

---

## 2. Bài toán class-incremental learning

Chuỗi học gồm các task:

$$
\mathcal T_1,\mathcal T_2,\ldots,\mathcal T_T,
$$

trong đó task $t$ cung cấp tập dữ liệu mới:

$$
\mathcal D_t=\{(x_i,y_i)\}_{i=1}^{N_t},
$$

và tập nhãn của các task không giao nhau:

$$
\mathcal Y_i\cap\mathcal Y_j=\varnothing,\qquad i\ne j.
$$

Sau task $t$, mô hình phải phân loại đồng thời mọi lớp đã gặp:

$$
\mathcal Y_{1:t}=\bigcup_{j=1}^{t}\mathcal Y_j,
$$

nhưng không được sử dụng lại ảnh huấn luyện của các task cũ. Hai yêu cầu đối
nghịch xuất hiện:

- **stability**: giữ được năng lực trên các lớp cũ;
- **plasticity**: thích nghi đủ tốt với các lớp mới.

Ký hiệu $f^t(x)\in\mathbb R^d$ là feature của ảnh $x$ sau khi adapter đã
được cập nhật ở task $t$. Trong dự án hiện tại, $d=768$.

---

## 3. RSIAT gốc: cấu trúc và ưu điểm

RSIAT sử dụng pretrained ViT và adapter để thích nghi tuần tự. Backbone chính
được đóng băng; phần adapter và classifier được học theo task. Kiến trúc có ba
ý tưởng chính.

### 3.1. Shared adapter

RSIAT duy trì một adapter dùng chung thay vì tạo một adapter riêng cho từng
task. Thiết kế này có các ưu điểm:

- số adapter ở inference không tăng tuyến tính theo số task;
- không cần biết task ID của ảnh kiểm thử;
- không cần router hoặc adapter retrieval;
- tránh retrieval error giữa các adapter;
- cách triển khai và inference đơn giản hơn multi-adapter methods.

### 3.2. Base representation steering

Ở task đầu, RSIAT vừa học cosine classifier vừa tổ chức lại không gian feature:

- kéo các mẫu cùng lớp lại gần;
- đẩy các mẫu khác lớp ra xa theo margin;
- tạo một nền biểu diễn ban đầu có cụm lớp rõ hơn.

Với feature đã chuẩn hóa $\hat z_i$, một dạng tương ứng với code hiện tại là:

$$
\mathcal L_{\mathrm{RS}}
=
\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\max(0,1-\hat z_i^\top\hat z_j)
+
\alpha
\frac{1}{|\mathcal N|}
\sum_{(i,j)\in\mathcal N}
\max(0,\hat z_i^\top\hat z_j-m_{\mathrm{RS}}),
$$

trong đó $\mathcal P$ và $\mathcal N$ lần lượt là tập cặp cùng lớp và khác
lớp trong mini-batch.

### 3.3. Representation alignment ở các task sau

Ở task $t>1$, RSIAT giữ một bản sao mạng cũ và quan sát hai biểu diễn của cùng
một ảnh mới:

$$
z_i^{t-1}=f^{t-1}(x_i),\qquad z_i^t=f^t(x_i),\quad x_i\in\mathcal D_t.
$$

Một residual projector học ánh xạ từ feature space cũ sang feature space mới:

$$
P_t(z_i^{t-1})\approx z_i^t.
$$

Điểm quan trọng là projector chỉ cần trong huấn luyện. Sau khi kết thúc task,
inference vẫn chỉ sử dụng ViT, shared adapter và classifier.

### 3.4. Exemplar-free classifier refinement

RSIAT không giữ ảnh cũ mà lưu thống kê feature theo lớp:

$$
\mu_c=\mathbb E[z\mid y=c],
\qquad
\Sigma_c=\operatorname{Cov}(z\mid y=c).
$$

Các synthetic features được lấy mẫu từ phân phối Gaussian để hiệu chỉnh lại
classifier trên toàn bộ lớp đã gặp. Phương pháp giữ được đặc tính
rehearsal-free ở mức dữ liệu ảnh và giảm class bias sau mỗi task.

### 3.5. Các ưu điểm được UMT-RSIAT kế thừa

UMT-RSIAT chủ đích giữ lại những điểm mạnh sau:

1. Một shared adapter xuyên suốt chuỗi task.
2. Backbone pretrained phần lớn được đóng băng.
3. Không lưu ảnh exemplar của lớp cũ.
4. Không cần task ID hoặc adapter retrieval khi inference.
5. Projector chỉ phục vụ learning/transport, không làm tăng inference model.
6. Classifier refinement bằng thống kê feature vẫn được duy trì.
7. Base representation steering ở task đầu vẫn được giữ để có baseline hình
   học ban đầu tương đương.

---

## 4. Những hạn chế quan trọng của RSIAT

### 4.1. Projector học trên lớp mới nhưng được kỳ vọng tổng quát sang lớp cũ

Tại task $t$, projector chỉ quan sát:

$$
\{(f^{t-1}(x),f^t(x)):x\in\mathcal D_t\}.
$$

Mọi ảnh này đều thuộc lớp mới. Tuy nhiên, projector sau đó được dùng để suy
luận sự dịch chuyển tại các vùng feature của lớp cũ. Điều này ngầm giả định:

$$
\text{drift trên miền lớp mới}
\approx
\text{drift trên miền lớp cũ}.
$$

Giả định có thể yếu khi lớp cũ và lớp mới khác nhau mạnh, feature drift mang
tính cục bộ, hoặc projector phi tuyến có capacity quá lớn và overfit vào task
hiện tại.

### 4.2. Point-wise MSE chưa trực tiếp bảo toàn phân phối

Alignment truyền thống chủ yếu giảm:

$$
\mathcal L_{\mathrm{point}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left\|P_t(z_i^{t-1})-z_i^t\right\|_2^2.
$$

MSE thấp trên các điểm của task mới không bảo đảm projector bảo toàn đúng:

- tâm của từng cụm lớp;
- độ phân tán theo các chiều feature;
- hình dạng phân phối dùng cho classifier refinement;
- biên quyết định giữa lớp cũ và lớp mới.

Đây là khoảng cách giữa mục tiêu huấn luyện projector và đối tượng mà
classifier refinement thực sự sử dụng: mean và covariance.

### 4.3. Residual autoencoder có thể phi tuyến quá mạnh

Một residual autoencoder nhiều tầng có khả năng biểu diễn lớn, nhưng dữ liệu để
học mỗi transition chỉ đến từ task hiện tại. Capacity lớn làm tăng nguy cơ:

- ghi nhớ drift của lớp mới;
- ngoại suy không ổn định quanh prototype cũ;
- khó phân tích mức biến dạng của feature space;
- tích lũy sai số qua nhiều transition.

Ngoài ra, nếu residual branch không được khởi tạo bằng zero, projector ban đầu
không thực sự là identity và có thể dịch feature ngay trước khi học được bằng
chứng về drift.

### 4.4. Hard separation/orthogonality quá đồng đều

Ý tưởng ép feature mới tách khỏi prototype cũ hỗ trợ stability, nhưng áp dụng
một ràng buộc giống nhau cho mọi cặp lớp bỏ qua cấu trúc của dữ liệu:

- một số lớp gần nghĩa nên chia sẻ một phần representation;
- phần lớn lớp cũ không phải đối thủ gây nhầm lẫn của một mẫu mới;
- ép tách tất cả cặp có thể cản trở plasticity;
- chi phí so sánh tăng theo số prototype cũ.

Đặc biệt, baseline code ban đầu lấy trung bình cosine similarity mà không lấy
trị tuyệt đối. Tối thiểu hóa đại lượng đó không tương đương chặt chẽ với việc
đưa cosine về 0; cosine âm lớn vẫn có thể làm loss nhỏ hơn. Vì vậy UMT-RSIAT
không tiếp tục dùng trực tiếp biểu thức này.

### 4.5. Prototype update và projector cần nhất quán

Trong baseline reproduction, projector tham gia alignment loss nhưng prototype
cũ lại có thể được cập nhật bằng một kernel-weighted displacement riêng. Khi
hai cơ chế không phải cùng một transition operator, khó trả lời:

- projector đang học phép chuyển nào;
- prototype được đưa vào feature space mới bằng phép chuyển nào;
- cải thiện hoặc sai số đến từ projector hay displacement heuristic.

Một hướng nghiên cứu rõ ràng hơn là dùng chính projector đã học để vận chuyển
thống kê lớp cũ.

### 4.6. Full covariance tốn bộ nhớ và không ổn định với ít mẫu

Với $d=768$, mỗi full covariance có $d^2=589{,}824$ phần tử. Bộ nhớ thống kê
tăng theo $O(Cd^2)$, trong đó $C$ là số lớp. Ước lượng full covariance cũng
có thể nhiễu hoặc gần suy biến khi số mẫu của lớp nhỏ hơn số chiều feature.

### 4.7. Sai số có thể tích lũy theo chuỗi task

Prototype sau nhiều task là kết quả của chuỗi transition ước lượng:

$$
\hat P_T\circ\hat P_{T-1}\circ\cdots\circ\hat P_2.
$$

Một sai số nhỏ ở mỗi transition có thể tích lũy và ảnh hưởng classifier
refinement. RSIAT cần được đánh giá không chỉ bằng final accuracy mà còn bằng
prototype drift, moment error và độ ổn định qua nhiều class order.

---

## 5. Hướng được lựa chọn: UMT-RSIAT

Hướng hiện được triển khai kết hợp ba thay đổi có liên hệ trực tiếp:

1. **Weakly nonlinear projector**: giới hạn độ phi tuyến và ưu tiên transition
   gần identity.
2. **Moment-aware alignment and transport**: căn chỉnh mean/variance và dùng
   cùng projector để vận chuyển thống kê lớp cũ.
3. **Adaptive top-K separation**: chỉ tách các prototype cũ gây nhầm lẫn và
   điều chỉnh mức tách theo độ liên quan trong feature space cũ.

Tên UMT nhấn mạnh mục tiêu chuyển từ point-wise alignment sang vận chuyển
thống kê phân phối. Trong code hiện tại, phần “uncertainty-aware” ở mức đầy đủ
chưa được triển khai; chưa có ensemble, Monte Carlo dropout confidence hoặc
confidence gate riêng cho mỗi prototype. Vì vậy không nên tuyên bố đây đã là
một uncertainty-aware method hoàn chỉnh.

---

## 6. Weakly nonlinear projector

### 6.1. Công thức

Projector tại transition $t-1\rightarrow t$ được định nghĩa:

$$
P_t(z)
=
z
+a_tL_t(z)
+b_tg_t(z),
$$

với nhánh tuyến tính low-rank:

$$
L_t(z)=U_t(V_tz),
$$

trong đó:

$$
V_t\in\mathbb R^{r\times d},
\qquad
U_t\in\mathbb R^{d\times r},
\qquad
r\ll d.
$$

Nhánh phi tuyến nhỏ là:

$$
g_t(z)=W_2\,\operatorname{GELU}(W_1z+b_1)+b_2.
$$

Hai gate được tham số hóa bằng sigmoid:

$$
a_t=\sigma(\theta_a),
\qquad
b_t=\sigma(\theta_b).
$$

Trong implementation hiện tại, $a_t$ và $b_t$ độc lập trong $(0,1)$,
không áp đặt $a_t+b_t=1$. Điều này cho phép projector tự chọn biên độ hai
nhánh mà không biến chúng thành một convex mixture cứng.

### 6.2. Khởi tạo identity

Trọng số của $U_t$ và lớp cuối $W_2$ được khởi tạo bằng zero:

$$
U_t=0,\qquad W_2=0,\qquad b_2=0.
$$

Do đó tại thời điểm tạo projector:

$$
P_t(z)=z.
$$

Đây là một prior phù hợp: trước khi quan sát dữ liệu, giả thuyết an toàn nhất là
feature space chưa dịch chuyển. Projector chỉ rời identity khi gradient cung
cấp bằng chứng về drift.

### 6.3. Tại sao dùng low-rank + nonlinear residual?

Nhánh low-rank tuyến tính nhằm học xu hướng drift toàn cục có khả năng ngoại
suy tương đối tốt. Số tham số của nhánh này xấp xỉ:

$$
2dr,
$$

thấp hơn nhiều so với phép biến đổi dense $d^2$ khi $r\ll d$.

Nhánh MLP nhỏ chỉ học phần drift mà linear operator không giải thích được. Gate
phi tuyến và regularization hạn chế khả năng MLP chi phối toàn bộ transition.
Thiết kế này đặt projector giữa hai cực:

- linear projector: dễ tổng quát nhưng có thể thiếu capacity;
- nonlinear RAE mạnh: linh hoạt nhưng dễ overfit task hiện tại.

### 6.4. Projector lifecycle

Với cấu hình hiện tại, một projector mới được tạo ở mỗi task $t>1$:

$$
P_2,P_3,\ldots,P_T.
$$

Mỗi projector chỉ mô hình hóa một transition liền kề. Sau khi thống kê cũ đã
được vận chuyển và checkpoint được lưu, projector hiện tại không tham gia
inference. Checkpoint giữ projector gần nhất để có thể resume đúng trạng thái
huấn luyện.

---

## 7. Moment-aware alignment

### 7.1. Point alignment

Với một mini-batch của task hiện tại:

$$
\tilde z_i^t=P_t(z_i^{t-1}),
$$

point loss là:

$$
\mathcal L_{\mathrm{point}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left\|\tilde z_i^t-z_i^t\right\|_2^2.
$$

Loss này giữ correspondence giữa cùng một ảnh trước và sau khi adapter thay
đổi.

### 7.2. Mean alignment theo lớp

Với lớp $k$ xuất hiện trong mini-batch, đặt:

$$
\mu_k^t
=
\frac{1}{|B_k|}\sum_{i\in B_k}z_i^t,
$$

$$
\tilde\mu_k^t
=
\frac{1}{|B_k|}\sum_{i\in B_k}\tilde z_i^t.
$$

Mean loss:

$$
\mathcal L_\mu
=
\frac{1}{|\mathcal K_B|}
\sum_{k\in\mathcal K_B}
\left\|\tilde\mu_k^t-\mu_k^t\right\|_2^2.
$$

Chỉ lớp có ít nhất hai mẫu trong batch được dùng cho class-wise moment. Nếu
không lớp nào đủ mẫu, code fallback sang moment của toàn mini-batch.

### 7.3. Diagonal variance alignment

Variance theo từng chiều của lớp $k$:

$$
v_k^t
=
\frac{1}{|B_k|}
\sum_{i\in B_k}(z_i^t-\mu_k^t)^{\odot2},
$$

$$
\tilde v_k^t
=
\frac{1}{|B_k|}
\sum_{i\in B_k}(\tilde z_i^t-\tilde\mu_k^t)^{\odot2}.
$$

Variance loss:

$$
\mathcal L_v
=
\frac{1}{|\mathcal K_B|}
\sum_{k\in\mathcal K_B}
\left\|\tilde v_k^t-v_k^t\right\|_2^2.
$$

UMT-RSIAT chọn diagonal variance thay vì full covariance trong alignment vì:

- chi phí giảm từ $O(d^2)$ xuống $O(d)$ cho mỗi lớp;
- estimate ổn định hơn với mini-batch nhỏ;
- không cần matrix square root hoặc eigendecomposition;
- phù hợp với mục tiêu chạy trên Colab/Kaggle có tài nguyên giới hạn.

### 7.4. Transport loss

Loss tổng của projector:

$$
\mathcal L_{\mathrm{transport}}
=
\lambda_p\mathcal L_{\mathrm{point}}
+\lambda_\mu\mathcal L_\mu
+\lambda_v\mathcal L_v.
$$

Trong cấu hình CIFAR-100 hiện tại:

$$
\lambda_p=1.0,
\qquad
\lambda_\mu=0.1,
\qquad
\lambda_v=0.01.
$$

Các giá trị này là cấu hình khởi đầu, chưa được xem là tối ưu nếu chưa có
ablation nhiều seed.

### 7.5. Mục đích

Moment alignment không thay thế point alignment mà bổ sung ràng buộc ở cấp
phân phối. Mục tiêu là:

- giảm trường hợp point MSE thấp nhưng cụm feature bị co/giãn sai;
- giúp projector phù hợp hơn với mean/variance được classifier refinement dùng;
- giảm sai lệch prototype sau nhiều task;
- tạo đại lượng chẩn đoán rõ ràng cho mean drift và variance drift.

---

## 8. Regularization giới hạn độ phi tuyến

Regularization hiện được triển khai:

$$
\mathcal L_{\mathrm{complex}}
=
b_t^2
+
\frac{1}{|W_2|}\|W_2\|_F^2.
$$

Mục đích:

- giữ nonlinear gate nhỏ nếu nhánh phi tuyến không thật sự cần thiết;
- giới hạn độ lớn lớp cuối của nonlinear residual;
- khuyến khích linear branch giải thích phần drift có cấu trúc đơn giản trước.

Trong tổng loss, thành phần này được nhân với $\lambda_c=0.001$ ở cấu hình
hiện tại.

Lưu ý: phiên bản hiện tại **chưa** triển khai Jacobian norm, spectral norm hay
Lipschitz constraint. Những đại lượng này là hướng ablation/mở rộng sau, không
phải thành phần đã có trong kết quả đang chạy.

---

## 9. Adaptive top-K separation

### 9.1. Chọn các prototype gây nhầm lẫn

Sau khi projector được áp dụng lên prototype cũ:

$$
\tilde p_c^t=P_t(p_c^{t-1}),
$$

độ tương tự giữa feature mới và prototype cũ là:

$$
s_{ic}
=
\cos(z_i^t,\tilde p_c^t)
=
\frac{(z_i^t)^\top\tilde p_c^t}
{\|z_i^t\|_2\|\tilde p_c^t\|_2}.
$$

Với mỗi mẫu $i$, chỉ lấy $K$ prototype cũ gần nhất:

$$
\mathcal N_K(i)
=
\operatorname{TopK}_{c\in\mathcal Y_{1:t-1}}s_{ic}.
$$

Điều này tập trung gradient vào các lớp cũ có khả năng gây nhầm lẫn thay vì
phạt đồng đều tất cả prototype.

### 9.2. Relatedness từ feature space cũ

Implementation hiện tại chưa dùng text/CLIP semantic. Relatedness được tính
trong feature space trước khi cập nhật adapter:

$$
r_{ic}
=
\frac{1+\cos(z_i^{t-1},p_c^{t-1})}{2}
\in[0,1].
$$

Hai tensor trong phép tính này được detach, nên relatedness đóng vai trò tín
hiệu tham chiếu, không tạo đường gradient phụ qua old network hay prototype.

### 9.3. Adaptive threshold

Ngưỡng cho phép của cặp $(i,c)$:

$$
\tau_{ic}
=
\tau_{\min}
+(\tau_{\max}-\tau_{\min})r_{ic}.
$$

Nếu lớp cũ và mẫu mới vốn gần nhau trong old feature space, $r_{ic}$ lớn và
ngưỡng $\tau_{ic}$ cao hơn. Mô hình được phép giữ một phần cấu trúc chung.
Nếu chúng ít liên quan, ngưỡng thấp hơn và separation mạnh hơn.

### 9.4. Hinge separation loss

$$
\mathcal L_{\mathrm{sep}}
=
\frac{1}{BK}
\sum_{i=1}^{B}
\sum_{c\in\mathcal N_K(i)}
\max(0,s_{ic}-\tau_{ic}).
$$

Chỉ cặp có cosine similarity vượt ngưỡng mới sinh gradient. Hai diagnostic
được log:

- `topk_sim`: cosine trung bình của các prototype được chọn;
- `active_sep`: tỷ lệ cặp top-K đang vi phạm threshold.

Trong cấu hình hiện tại:

$$
K=10,
\qquad
\tau_{\min}=0.1,
\qquad
\tau_{\max}=0.5,
\qquad
\lambda_{\mathrm{sep}}=0.4.
$$

### 9.5. Mục đích

Adaptive top-K separation hướng đến:

- giảm handbrake effect của hard orthogonality;
- giữ plasticity cho lớp mới;
- bảo toàn phần representation hợp lý giữa các lớp gần nhau;
- tập trung vào hard old classes;
- giảm phần loss cần xử lý từ toàn bộ lớp cũ xuống $K$ lớp được chọn sau
  bước tính similarity.

Lưu ý về độ phức tạp: implementation hiện vẫn tính ma trận similarity với toàn
bộ prototype để tìm top-K, nên phép tìm kiếm chính xác vẫn có chi phí
$O(BC_{\mathrm{old}}d)$. Lợi ích hiện tại chủ yếu là sparsify loss/gradient,
chưa giảm triệt để chi phí retrieval. FAISS hoặc approximate nearest-neighbor
index mới là bước cần thiết nếu mở rộng đến hàng chục nghìn lớp.

---

## 10. Mục tiêu huấn luyện tổng thể

### 10.1. Task đầu

$$
\mathcal L^{(1)}
=
\mathcal L_{\mathrm{cos}}
+\lambda_{\mathrm{RS}}(e)\mathcal L_{\mathrm{RS}},
$$

trong đó $\lambda_{\mathrm{RS}}(e)$ được warm-up theo epoch.

### 10.2. Các task sau

$$
\boxed{
\mathcal L^{(t)}
=
\mathcal L_{\mathrm{cos}}
+\lambda_p\mathcal L_{\mathrm{point}}
+\lambda_\mu\mathcal L_\mu
+\lambda_v\mathcal L_v
+\lambda_{\mathrm{sep}}\mathcal L_{\mathrm{sep}}
+\lambda_c\mathcal L_{\mathrm{complex}}
}
$$

Trong cấu hình CIFAR-100 đang chạy:

| Thành phần | Trọng số |
|---|---:|
| Point alignment | 1.0 |
| Mean alignment | 0.1 |
| Diagonal variance alignment | 0.01 |
| Adaptive top-K separation | 0.4 |
| Complexity regularization | 0.001 |

Cosine classification loss chỉ được tính trên classifier head của các lớp mới
trong stage huấn luyện task. Sau đó classifier alignment sử dụng synthetic
features của mọi lớp để hiệu chỉnh các head.

---

## 11. Vận chuyển prototype và variance

### 11.1. Lý do không chỉ biến đổi prototype mean

Với projector phi tuyến:

$$
P_t(\mathbb E[z])\ne\mathbb E[P_t(z)]
$$

nói chung. Do đó chỉ tính $P_t(\mu_c)$ có thể không cho mean đúng của phân
phối sau biến đổi, đồng thời không cho biết variance mới.

### 11.2. Diagonal Monte Carlo transport

Với mỗi lớp cũ $c$, lưu:

$$
\mu_c^{t-1},
\qquad
v_c^{t-1}=\operatorname{diag}(\Sigma_c^{t-1}).
$$

Sinh $M$ feature:

$$
\epsilon_j\sim\mathcal N(0,I),
$$

$$
z_{c,j}^{t-1}
=
\mu_c^{t-1}
+\sqrt{v_c^{t-1}}\odot\epsilon_j.
$$

Vận chuyển từng sample:

$$
\tilde z_{c,j}^{t}=P_t(z_{c,j}^{t-1}).
$$

Ước lượng thống kê mới:

$$
\mu_c^t
=
\frac{1}{M}\sum_{j=1}^{M}\tilde z_{c,j}^{t},
$$

$$
v_c^t
=
\frac{1}{M}\sum_{j=1}^{M}
(\tilde z_{c,j}^{t}-\mu_c^t)^{\odot2}.
$$

Sau đó:

$$
\Sigma_c^t=\operatorname{diag}(v_c^t).
$$

Trong full config, $M=128$. Một epsilon nhỏ được dùng để bảo đảm variance
dương và tránh lỗi khi tạo Gaussian.

### 11.3. Tính nhất quán của transition

Projector được dùng cho cả:

1. alignment giữa old/current features;
2. prototype được tham chiếu trong separation;
3. vận chuyển mean/variance của lớp cũ.

Điều này tạo một transition operator thống nhất và giúp phân tích đóng góp dễ
hơn so với việc alignment dùng projector nhưng prototype update dùng một
heuristic độc lập.

### 11.4. Compact checkpoint

Trong runtime, covariance vẫn được biểu diễn dưới dạng ma trận đường chéo để
tương thích với classifier refinement hiện có. Khi lưu checkpoint, chỉ vector
variance được lưu:

$$
\text{storage}=O(Cd)
$$

thay vì:

$$
O(Cd^2).
$$

Khi resume, vector variance được khôi phục bằng `diag_embed` thành ma trận
đường chéo.

---

## 12. Luồng hoạt động theo một incremental task

Với task $t>1$, quy trình đang triển khai là:

1. Giữ bản sao $f^{t-1}$ của network sau task trước ở trạng thái frozen.
2. Tạo projector $P_t$ mới, khởi tạo identity.
3. Với ảnh mới $x\in\mathcal D_t$, lấy $z^{t-1}=f^{t-1}(x)$ và
   $z^t=f^t(x)$.
4. Tối ưu cosine classification loss trên lớp mới.
5. Tối ưu point, mean và diagonal-variance alignment giữa $P_t(z^{t-1})$
   và $z^t$.
6. Chọn top-K prototype cũ gây nhầm lẫn và áp dụng adaptive separation.
7. Regularize nonlinear capacity của projector.
8. Sau khi stage huấn luyện kết thúc, dùng $P_t$ để vận chuyển thống kê của
   toàn bộ lớp cũ bằng diagonal Monte Carlo transport.
9. Tính mean/covariance thật cho các lớp mới từ dữ liệu task hiện tại.
10. Sinh synthetic features từ thống kê mọi lớp để classifier refinement.
11. Đánh giá trên toàn bộ lớp đã gặp.
12. Lưu network, projector gần nhất, mean/variance, class order, task sizes và
    accuracy curves vào checkpoint.
13. Sao chép network hiện tại thành frozen old network cho task kế tiếp.

---

## 13. Giả thuyết nghiên cứu cần kiểm chứng

### H1 — Khả năng tổng quát của transition

Weakly nonlinear projector sẽ có sai số trên old-class oracle features thấp hơn
RAE mạnh, dù training alignment error trên task hiện tại có thể tương đương
hoặc cao hơn một chút.

### H2 — Bảo toàn phân phối

Moment-aware alignment sẽ giảm mean error và variance error của prototype sau
transport, từ đó tăng final accuracy và giảm forgetting.

### H3 — Stability–plasticity tốt hơn

Adaptive top-K separation sẽ giữ old-class accuracy cạnh tranh trong khi tăng
new-class accuracy so với hard/full separation.

### H4 — Hiệu quả bộ nhớ

Diagonal statistics và compact checkpoint sẽ giảm mạnh bộ nhớ thống kê mà
không làm accuracy suy giảm đáng kể.

### H5 — Tích lũy drift chậm hơn

Identity initialization, low-rank prior và moment constraints sẽ làm prototype
error tăng chậm hơn theo số task.

Các giả thuyết trên chưa được xác nhận chỉ bằng việc pipeline chạy không lỗi.
Smoke test chỉ chứng minh implementation có thể đi qua toàn bộ task và các loss
có giá trị hữu hạn.

---

## 14. Những phần quan trọng trong repository

| Thành phần | File | Vai trò |
|---|---|---|
| Weakly nonlinear projector | `models/projectors.py` | Identity, low-rank và nonlinear branches, gates, initialization và regularizer |
| Incremental objective | `models/RSIAT_adapter.py` | Ghép classification, transport, separation và complexity loss |
| Statistics transport | `models/RSIAT_adapter.py` | Monte Carlo transport mean/diagonal variance sau mỗi task |
| Moment và top-K losses | `utils/research_losses.py` | Class-wise moments, adaptive thresholds và hinge loss |
| Class statistics/refinement | `models/base.py` | Tính thống kê lớp, Gaussian sampling, classifier alignment, checkpoint |
| Backbone/shared adapter | `network/vision_transformer_adapter.py` | ViT-B/16 pretrained và AdaptFormer-style shared adapters |
| Incremental classifier | `network/classifier.py` | Cosine-normalized task heads |
| Task protocol | `data/data_manager.py` | Class order, task increments và tập train/test theo task |
| Training/resume | `trainer.py` | Seed, checkpoint path, task loop, resume và giới hạn task mỗi phiên |
| Full CIFAR config | `exps/umt_adapter_cifar224.json` | Hyperparameters của full run seed 1993 |
| Colab workflow | `RSIAT_UMT_Colab.ipynb` | Persistent dataset/checkpoint trên Google Drive |
| Kaggle workflow | `RSIAT_UMT_Kaggle.ipynb` | Resume theo task bằng Kaggle Input/Output |

---

## 15. Giới hạn của phiên bản UMT-RSIAT hiện tại

Tài liệu nghiên cứu cần nêu rõ những giới hạn sau:

1. Projector vẫn chỉ được fit bằng ảnh lớp mới; weak nonlinearity giảm rủi ro
   nhưng không loại bỏ domain extrapolation problem.
2. Moment loss sử dụng mini-batch estimates, có thể nhiễu nếu mỗi lớp có ít
   mẫu trong batch.
3. Diagonal variance bỏ qua tương quan giữa các chiều feature.
4. Gaussian đơn không biểu diễn tốt lớp đa mode.
5. Monte Carlo transport có sampling noise và chi phí tăng theo số lớp cũ cùng
   số sample $M$.
6. Relatedness là feature-aware, chưa phải semantic-aware từ class name hoặc
   CLIP text embedding.
7. Exact top-K vẫn cần tính similarity với toàn bộ prototype cũ.
8. Chưa có uncertainty score riêng để quyết định mức cập nhật cho từng
   prototype.
9. Chưa có Jacobian/spectral constraint trực tiếp cho projector.
10. Thực nghiệm hiện tập trung class-disjoint, task-aware training boundary và
    closed-set evaluation; chưa chứng minh task-free, online hay open-world.
11. Full run seed 1993 riêng lẻ chưa đủ để kết luận tính ổn định thống kê; cần
    ít nhất ba class orders nếu dùng cho báo cáo nghiên cứu.

---

## 16. Ranh giới giữa phần đã triển khai và phần mở rộng tương lai

### Đã triển khai

- shared ViT adapter và incremental cosine classifier;
- weakly nonlinear projector với low-rank branch;
- exact identity initialization;
- projector mới cho mỗi transition;
- point + class-wise mean + diagonal variance alignment;
- feature-aware adaptive top-K hinge separation;
- projector-based diagonal Monte Carlo statistics transport;
- compact diagonal checkpoints;
- seed-isolated resume trên Colab/Kaggle;
- logging từng loss component.

### Chưa triển khai

- CLIP/text-semantic adaptive threshold;
- uncertainty-aware soft update cho từng prototype;
- mixture of prototypes/Gaussians;
- shared low-rank covariance basis;
- Jacobian, spectral hoặc explicit Lipschitz regularization;
- approximate nearest-neighbor retrieval;
- oracle old-feature drift diagnostics tự động;
- đầy đủ forgetting/intransigence metrics và statistical significance tests.

Việc phân biệt hai nhóm này rất quan trọng khi viết báo cáo: không nên mô tả
một ý tưởng tương lai như một thành phần đã đóng góp hoặc đã được thực nghiệm.

---

## 17. Danh sách bằng chứng cần thu thập và trực quan hóa

Log full run task 0–9 của seed 1993 đã có và được phân tích ở Mục 18. Các run
tiếp theo vẫn cần trích tối thiểu:

- top-1 và top-5 curve theo task;
- average accuracy và final accuracy;
- old/new accuracy theo task;
- point, mean, variance, separation và complexity loss;
- `topk_sim` và `active_sep`;
- thời gian train, classifier alignment và evaluation theo task;
- số tham số trainable;
- checkpoint/statistics memory;
- baseline RSIAT chạy cùng seed 1993, class order và epoch budget.

Các biểu đồ phù hợp sau này gồm:

1. Accuracy theo số task: RSIAT và UMT-RSIAT.
2. Old/new accuracy theo task để thể hiện stability–plasticity.
3. Từng thành phần loss theo task.
4. `active_sep` và `topk_sim` theo số lớp cũ.
5. Forgetting theo nhóm task/lớp.
6. Runtime và checkpoint size theo số task.
7. Ablation projector/moment/top-K nếu có đủ run.

Sau một seed, kết luận đúng nhất vẫn là: **UMT-RSIAT là một giả thuyết cải tiến
có cơ sở và đã chạy được toàn bộ pipeline, nhưng cấu hình hiện tại chưa cải
thiện RSIAT gốc và chưa đủ bằng chứng để kết luận về hiệu quả thống kê.**

---

## 18. Phân tích full run CIFAR-100, seed 1993

### 18.1. Phạm vi và tính hợp lệ của phép so sánh

Kết quả UMT được lấy từ log:

`logs/umt_adapter/cifar224/0/10/umt_weak_moment_topk_1993_pretrained_vit_b16_224_in21k_adapter (1).log`

Baseline được lấy từ run hoàn chỉnh cuối cùng trong log:

`logs/adapter/cifar224/0/10/all_1993_pretrained_vit_b16_224_in21k_adapter.log`

Hai run dùng cùng CIFAR-100 B0I10, seed và class order 1993, ViT-B/16 IN21K,
batch size 64, 10 epoch cho task đầu, 30 epoch cho mỗi task tăng dần và 10 epoch
classifier alignment. Vì file baseline chứa nhiều lần chạy nối tiếp nhau, phép
so sánh dưới đây chỉ dùng chuỗi full run cuối cùng có đủ task 0–9; không trộn
với các smoke run trước đó.

Khác biệt quan trọng thuộc chính phương pháp: baseline dùng `ssca=true` và cơ
chế RSIAT gốc, còn UMT dùng `ssca=false`, projector phi tuyến yếu, moment loss,
adaptive top-K và diagonal Monte Carlo statistics transport. Do đó đây là so
sánh hai pipeline hoàn chỉnh, chưa phải ablation cô lập từng thành phần.

Log UMT bị ngắt phiên Colab ở một số task nhưng resume đã nạp đúng checkpoint
gần nhất và cuối cùng lưu được `task_9.pkl`. Log có đủ đúng 10 kết quả đánh giá
hoàn tất theo thứ tự task 0–9. Tổng thời gian của các đoạn chạy thành công xấp
xỉ 15 giờ 53 phút, không tính thời gian chờ giữa các phiên và các đoạn bị ngắt.

### 18.2. Kết quả accuracy theo task

| Task | Số lớp đã học | UMT top-1 (%) | RSIAT top-1 (%) | UMT - RSIAT |
|---:|---:|---:|---:|---:|
| 0 | 10 | 99.00 | 99.00 | 0.00 |
| 1 | 20 | 97.30 | 97.60 | -0.30 |
| 2 | 30 | 96.40 | 96.90 | -0.50 |
| 3 | 40 | 95.60 | 96.18 | -0.58 |
| 4 | 50 | 94.56 | 95.22 | -0.66 |
| 5 | 60 | 93.65 | 94.32 | -0.67 |
| 6 | 70 | 93.63 | 94.10 | -0.47 |
| 7 | 80 | 92.08 | 92.58 | -0.50 |
| 8 | 90 | 91.69 | 92.36 | -0.67 |
| 9 | 100 | 91.41 | 92.17 | -0.76 |

Các chỉ số tổng hợp:

| Chỉ số | UMT-RSIAT | RSIAT gốc | Chênh lệch UMT - RSIAT |
|---|---:|---:|---:|
| Average incremental top-1 | 94.532 | 95.043 | -0.511 |
| Final top-1 | 91.41 | 92.17 | -0.76 |
| Average incremental top-5 | 99.485 | 99.547 | -0.062 |
| Final top-5 | 99.19 | 99.27 | -0.08 |
| Sụt giảm top-1 từ task 0 đến task 9 | 7.59 | 6.83 | +0.76 |

UMT thấp hơn baseline từ task 1 và khoảng cách nhìn chung tăng khi số task tăng.
Vì vậy run này **không xác nhận giả thuyết UMT cải thiện accuracy hoặc giảm
forgetting của RSIAT**. Top-5 gần như giữ nguyên, trong khi top-1 giảm rõ hơn;
điều này gợi ý nhiều lỗi nằm ở việc xếp hạng nhầm giữa các lớp gần nhau, thay vì
đưa lớp đúng ra hoàn toàn khỏi nhóm ứng viên đầu.

### 18.3. Stability–plasticity và forgetting theo nhóm lớp

Ở task cuối, UMT đạt `old=91.08%` và `new=94.40%`. Khả năng học lớp mới vẫn tốt,
nhưng độ chính xác lớp cũ thấp hơn lớp mới 3.32 điểm.

Trung bình trên các task tăng dần 1–9 cho thấy trade-off rất rõ:

| Thành phần accuracy | UMT-RSIAT | RSIAT gốc | UMT - RSIAT |
|---|---:|---:|---:|
| Old classes | 93.609 | 94.660 | -1.051 |
| New classes | 96.233 | 94.544 | +1.689 |

UMT cao hơn baseline ở `new` tại cả 9/9 task tăng dần, nhưng thấp hơn ở `old`
cũng tại cả 9/9 task. Nói cách khác, phương pháp hiện tăng plasticity một cách
nhất quán nhưng đánh đổi stability quá nhiều, làm total accuracy cuối cùng giảm.

Kết quả cuối theo từng nhóm 10 lớp là:

| Nhóm lớp | 00–09 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 | 90–99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UMT | 89.8 | 90.6 | 94.9 | 92.0 | 88.6 | 87.1 | 92.5 | 89.2 | 95.0 | 94.4 |
| RSIAT | 91.2 | 92.0 | 94.6 | 93.6 | 89.9 | 89.8 | 93.1 | 90.3 | 93.7 | 93.5 |
| UMT - RSIAT | -1.4 | -1.4 | +0.3 | -1.6 | -1.3 | -2.7 | -0.6 | -1.1 | +1.3 | +0.9 |

Với mỗi nhóm lớp cũ, lấy accuracy cao nhất từng đạt được trừ accuracy ở task 9,
average group forgetting trên chín nhóm cũ là **5.67 điểm** cho UMT và **3.32
điểm** cho RSIAT. Đây là chẩn đoán quan trọng nhất: cấu hình UMT hiện tại tăng
plasticity ở một vài nhóm muộn, nhưng chưa bảo vệ stability của các nhóm cũ;
nhóm 50–59 giảm mạnh nhất và kết thúc thấp hơn baseline 2.7 điểm.

Chỉ số trên là group-level forgetting tính lại từ các dòng `CNN` trong log,
không phải class-level forgetting chuẩn hóa sẵn bởi code. Nó phù hợp để chẩn
đoán run này, nhưng báo cáo chính thức nên bổ sung forgetting theo từng lớp và
trung bình qua nhiều seed.

### 18.4. Projector, moment loss và adaptive top-K đang hoạt động ra sao

Ở epoch cuối của task 1, `active_sep=0.0306` và `topk_sim=0.1793`. Hai đại lượng
này tăng dần, đạt đỉnh lần lượt 0.1723 và 0.2987 ở task 8, rồi kết thúc ở 0.1342
và 0.2834 tại task 9. Như vậy adaptive top-K có thực sự kích hoạt và phát hiện
nhiều cặp old–new khó hơn khi số lớp tăng; nó không bị vô hiệu hoàn toàn.

Tại task 9, các loss thô được log là:

$$
L_{point}=0.0204,\quad L_{mean}=0.0063,\quad L_{var}=0.0576,\quad
L_{sep}=0.0082,\quad L_{complexity}=0.0100.
$$

Sau khi nhân các trọng số cấu hình, phần regularization xấp xỉ:

$$
0.0204 + 0.1(0.0063) + 0.01(0.0576)
+ 0.4(0.0082) + 0.001(0.0100) = 0.024896,
$$

khớp với `Losses_rt=0.025` sau làm tròn. Point alignment chiếm khoảng 81.9%,
separation 13.2%, mean 2.5%, variance 2.3% và complexity dưới 0.1% tổng phần
regularization ở task cuối. Cấu hình vì thế vẫn bị point loss chi phối; moment
constraints, đặc biệt variance, có ảnh hưởng gradient tương đối nhỏ.

`complexity` được in là 0.0100 ở epoch cuối của mọi task. Vì giá trị này trùng
với bình phương `nonlinear_gate_init=0.1`, có hai khả năng cần kiểm tra bằng log
chi tiết hơn: nhánh phi tuyến được dùng rất yếu, hoặc thay đổi của gate bị che
bởi độ chính xác bốn chữ số. Chưa thể kết luận projector thật sự gần tuyến tính
nếu chưa log trực tiếp gate và tỷ lệ chuẩn residual
$\lVert r(z)\rVert_2/\lVert z\rVert_2$.

Projector làm tăng 131,394 tham số trong giai đoạn train của task tăng dần,
xấp xỉ 0.15% số tham số mạng. Chi phí tham số không phải vấn đề chính quan sát
được; vấn đề hiện tại là hiệu quả giữ tri thức cũ.

### 18.5. Ảnh hưởng của classifier alignment

Lấy `Test_accy` tại epoch 30 làm kết quả trước CA và `CNN total` làm kết quả sau
CA, mức thay đổi ở task 1–9 lần lượt là:

$$
[0.00,\ +0.67,\ +1.20,\ +0.30,\ -0.07,\ +0.32,\ +0.58,\ +0.49,\ +0.21].
$$

CA cải thiện trung bình khoảng 0.41 điểm trên chín task tăng dần và chỉ giảm nhẹ
0.07 điểm ở task 5. Vì vậy classifier alignment vẫn hữu ích, nhưng mức tăng này
không đủ bù phần forgetting lớn hơn baseline.

### 18.6. Kết luận nghiên cứu sau run đầu tiên

Run này chứng minh ba điểm tích cực:

1. Pipeline UMT hoàn chỉnh, checkpoint compact và resume qua nhiều phiên Colab
   đều hoạt động đến task 9.
2. Adaptive top-K thực sự phản ứng khi số lớp và mức giao thoa tăng.
3. UMT giữ plasticity tốt cho các lớp mới và có overhead tham số nhỏ.

Tuy nhiên, mục tiêu chính là cải thiện stability của RSIAT chưa đạt ở cấu hình
hiện tại. Chênh lệch âm xuất hiện nhất quán ở 9/9 task tăng dần, final accuracy
thấp hơn 0.76 điểm và group forgetting cao hơn 2.35 điểm. Đây không giống một
dao động đơn lẻ ở task cuối, dù vẫn chưa thể khẳng định ý nghĩa thống kê với chỉ
một seed.

### 18.7. Thứ tự thí nghiệm tiếp theo

Không nên chạy ngay nhiều seed cho đúng cấu hình hiện tại vì sẽ tốn thời gian
để xác nhận lại một cấu hình đang thua baseline. Nên ưu tiên ablation ngắn nhưng
có kiểm soát, giữ nguyên seed 1993 và mọi training budget:

1. **Transport-only:** projector + statistics transport, tắt moment và top-K.
2. **Transport + moment:** bật point/mean/variance, tắt top-K.
3. **Transport + top-K:** bật separation, tắt mean/variance.
4. **Full UMT với stability mạnh hơn:** thử tăng `mean_weight` và
   `variance_weight` theo lưới nhỏ; đồng thời theo dõi old accuracy, không chỉ
   total accuracy.
5. Log `linear_gate`, `nonlinear_gate`, residual ratio, gradient norm từng loss
   và prototype drift theo task để biết projector có học đúng cơ chế dự kiến.
6. Chỉ sau khi chọn được cấu hình không thua baseline ở seed 1993 mới chạy thêm
   ít nhất hai seed/class order và báo cáo mean ± standard deviation.

Nếu ablation cho thấy top-K làm giảm old accuracy, cần thử threshold/top-K mềm
hơn hoặc schedule `separation_weight` theo task. Nếu transport-only đã gây giảm,
trọng tâm phải chuyển sang độ chính xác của diagonal Monte Carlo transport hoặc
giữ lại shared low-rank covariance thay vì tiếp tục tăng separation loss.
