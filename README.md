# Báo Cáo Phân Loại Tidal Disruption Events (TDE) với Machine Learning

## 1. Giới thiệu

### 1.1. Bối cảnh bài toán
Tidal Disruption Events (TDE) là hiện tượng thiên văn xảy ra khi một ngôi sao bị xé toán bởi lực triều của lỗ đen siêu khối lượng. Việc phát hiện TDE rất quan trọng trong nghiên cứu vật lý học lỗ đen và động lực học thiên hà.

### 1.2. Mục tiêu dự án
- Xây dựng mô hình phân loại tự động để phát hiện TDE từ dữ liệu đường cong ánh sáng (light curves)
- Tối ưu hóa F1-Score để cân bằng giữa Precision và Recall trên bộ dữ liệu không cân bằng
- Sử dụng phương pháp ensemble learning kết hợp LightGBM và XGBoost

### 1.3. Thách thức
- **Dữ liệu không cân bằng nghiêm trọng**: Tỷ lệ TDE chỉ chiếm 4.86% trong tập huấn luyện (148/3043 mẫu)
- **Dữ liệu time-series phức tạp**: Đường cong ánh sáng qua nhiều băng tần khác nhau
- **Đặc trưng vật lý đặc thù**: Cần hiểu biết về đặc điểm vật lý của TDE

---

## 2. Dữ liệu (Dataset)

### 2.1. Cấu trúc dữ liệu
**Tập huấn luyện**: 3,043 đối tượng
- Non-TDE: 2,895 mẫu (95.14%)
- TDE: 148 mẫu (4.86%)

**Tập kiểm tra**: 7,135 đối tượng

**Phân chia dữ liệu**: 20 splits để tổ chức dữ liệu đường cong ánh sáng

### 2.2. Đặc điểm đường cong ánh sáng
- **Các băng tần quan sát**: u, g, r, i, z, y (6 filters)
- **Thông tin mỗi quan sát**:
  - Time (MJD): Thời gian quan sát
  - Flux: Cường độ sáng
  - Flux_err: Sai số đo
  - Filter: Băng tần quan sát

### 2.3. Metadata
- **Z**: Redshift (độ dịch chuyển đỏ)
- **EBV**: Extinction parameter (độ tắt sáng do bụi)

---

## 3. Phương pháp tiếp cận

### 3.1. Kiến trúc tổng thể

```
Dữ liệu đầu vào (Light curves)
          ↓
Feature Engineering (132 features)
          ↓
Feature Selection (120 features)
          ↓
Hyperparameter Tuning (Optuna - 50 trials)
          ↓
Stacking Ensemble (LightGBM 60% + XGBoost 40%)
          ↓
Threshold Optimization (maximize F1)
          ↓
Dự đoán cuối cùng
```

### 3.2. Trích xuất đặc trưng (Feature Engineering)

#### 3.2.1. Nhóm đặc trưng toàn cục (Global Features)
- **Thống kê cơ bản**: n_obs, duration, n_filters
- **Cadence (tần suất quan sát)**: mean, std, median của khoảng thời gian giữa các quan sát
- **Thống kê Flux**: mean, std, median, max, min, range, IQR, MAD
- **Signal-to-Noise Ratio (SNR)**: mean, median, max
- **Detection rate**: Tỷ lệ quan sát có Flux > 3σ

#### 3.2.2. Nhóm đặc trưng theo băng tần (Per-filter Features)
Cho mỗi filter (u, g, r, i, z, y):
- Số lượng quan sát
- Thống kê Flux: mean, std, max, min, range
- **Variability indices**:
  - Stetson J: Chỉ số biến đổi chuẩn hóa
  - Excess variance: Phương sai vượt trội
- **Peak analysis**:
  - Time to peak: Thời gian lên đỉnh
  - Time from peak: Thời gian xuống từ đỉnh
  - Rise-fall ratio: Tỷ số thời gian lên/xuống

#### 3.2.3. Nhóm đặc trưng đa băng tần (Multi-band Features)
- **Color indices**: Chênh lệch flux giữa các băng tần (u-g, g-r, r-i, i-z, u-r)
- **Peak synchronization**: Độ phân tán thời gian đỉnh giữa các filter

#### 3.2.4. Nhóm đặc trưng đặc thù TDE (TDE-specific Features)

**A. Parabolic fits**
- Fit polynomial bậc 2 cho pha tăng và giảm sáng
- TDE có xu hướng có đường cong đối xứng hơn AGN

**B. Power-law signatures** (Đặc trưng vật lý quan trọng!)
Lý thuyết vật lý TDE:
- **Pha tăng sáng**: F(t) ∝ t^(5/3) → slope ≈ 1.67 trong log-log scale
- **Pha giảm sáng**: F(t) ∝ t^(-5/3) → slope ≈ -1.67 trong log-log scale

Các đặc trưng được tính:
- `{filter}_rise_powerlaw`: Hệ số góc pha tăng
- `{filter}_decline_powerlaw`: Hệ số góc pha giảm
- `{filter}_rise_powerlaw_r2`: R² của fit pha tăng
- `{filter}_decline_powerlaw_r2`: R² của fit pha giảm

**C. Peak counting**
- Số lượng đỉnh trong đường cong ánh sáng
- TDE: thường 1 đỉnh chính
- AGN: nhiều đỉnh biến động

#### 3.2.5. Đặc trưng thống kê nâng cao
- **Skewness**: Độ lệch phân phối
- **Kurtosis**: Độ nhọn phân phối
- **Percentiles**: P05, P95, P10/P90 ratio
- **Beyond 1σ**: Tỷ lệ điểm nằm ngoài 1 độ lệch chuẩn

#### 3.2.6. Interaction Features (Feature engineering nâng cao)
Các đặc trưng tổng hợp domain-specific:
- **Smoothness score**: detection_rate / flux_MAD
- **Symmetry score**: Đo độ đối xứng quanh đỉnh
- **Parabolic quality**: Trung bình R² của fit parabolic
- **TDE power-law score**: Đo độ phù hợp với power-law lý thuyết
  ```
  score = 1 / (1 + |slope_rise - 1.67| + |slope_decline + 1.67|)
  ```
- **Peak coherence**: Đồng bộ đỉnh giữa các băng tần
- **Variability contrast**: Chênh lệch biến động giữa các filter
- **Quality score**: SNR × detection_rate

**Tổng số features ban đầu**: 132 features

### 3.3. Lựa chọn đặc trưng (Feature Selection)

#### 3.3.1. Phương pháp kết hợp
1. **Mutual Information (MI)**: Đo lượng thông tin chung giữa feature và target
2. **Random Forest Importance**: Đo độ quan trọng từ RF với 100 trees
3. **Combined Score**: 
   ```
   score = 0.6 × MI_normalized + 0.4 × RF_normalized
   ```

#### 3.3.2. Kết quả
- **Số features được chọn**: TOP 120/132
- **Top 3 features quan trọng nhất**:
  1. `snr_median` (0.9305): SNR trung vị
  2. `r_excess_var` (0.7999): Excess variance băng r
  3. `r_decline_powerlaw` (0.7314): Power-law slope pha giảm băng r

### 3.4. Tối ưu siêu tham số (Hyperparameter Tuning)

#### 3.4.1. Framework: Optuna
- **Objective**: Maximize F1-Score (cross-validation 5-fold)
- **Number of trials**: 50
- **Search space**:
  - num_leaves: [20, 80]
  - learning_rate: [0.01, 0.1] (log-scale)
  - feature_fraction: [0.5, 0.9]
  - bagging_fraction: [0.5, 0.9]
  - min_child_samples: [5, 50]
  - max_depth: [4, 12]
  - reg_alpha, reg_lambda: [0.01, 10] (log-scale)

### 3.5. Mô hình Ensemble

#### 3.5.1. Kiến trúc Stacking
```
         Training Data
              ↓
        5-Fold CV
       /         \
  LightGBM    XGBoost
      ↓           ↓
   OOF_lgb   OOF_xgb
       \         /
        Weighted Average
    (0.6 LGB + 0.4 XGB)
              ↓
    Threshold Optimization
              ↓
       Final Prediction
```

#### 3.5.2. Chi tiết mô hình

**LightGBM Configuration**:
- Boosting type: GBDT (Gradient Boosting Decision Tree)
- Objective: Binary classification
- Metric: Binary log-loss
- Early stopping: 50 rounds
- Max boosting rounds: 2000

**XGBoost Configuration**

#### 3.5.3. Ensemble strategy
- **Weighted average**: LightGBM (60%) + XGBoost (40%)
- **Rationale**: 
  - LightGBM có hiệu suất ổn định hơn trên fold validation
  - XGBoost bổ sung diversity cho ensemble
  - Tỷ lệ 60-40 được chọn dựa trên cross-validation performance

### 3.6. Tối ưu ngưỡng (Threshold Optimization)

#### 3.6.1. Phương pháp
- **Search space**: [0.05, 0.95] với bước nhảy 0.01
- **Metric**: F1-Score
- **Strategy**: Grid search trên OOF predictions

#### 3.6.2. Kết quả
- **Optimal threshold**: 0.300
- **F1-Score at optimal threshold**: 0.5199
- **Default threshold (0.5) sẽ không tối ưu** do class imbalance

### 3.7. Xử lý Class Imbalance

#### 3.7.1. Các kỹ thuật áp dụng
1. **Scale_pos_weight**: 
   - Ratio = 2895/148 = 19.56
   - Tăng trọng số cho class TDE (thiểu số)

2. **Threshold tuning**: 
   - Giảm threshold xuống 0.3 thay vì 0.5
   - Tăng recall cho class TDE

3. **StratifiedKFold**: 
   - Đảm bảo tỷ lệ TDE/Non-TDE giống nhau ở mỗi fold
   - 5-fold validation

4. **Evaluation metric**:
   - Tối ưu F1-Score thay vì Accuracy
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## 4. Kết quả (Results)
#### Top 20 Features quan trọng nhất (từ LightGBM)

| Rank | Feature | Importance | Loại |
|------|---------|------------|------|
| 1 | i_excess_var | 11306.02 | Variability (filter i) |
| 2 | i_rise_powerlaw_r2 | 3468.75 | TDE physics |
| 3 | z_stetson_j | 3452.74 | Variability (filter z) |
| 4 | flux_range | 2969.16 | Global stats |
| 5 | y_flux_std | 2567.22 | Filter y stats |
| 6 | g_excess_var | 2004.12 | Variability (filter g) |
| 7 | snr_median | 1878.23 | SNR |
| 8 | g_flux_mean | 1877.68 | Filter g stats |
| 9 | flux_iqr | 1851.64 | Global stats |
| 10 | u_flux_range | 1804.66 | Filter u stats |
| 11 | n_filters | 1772.71 | Global |
| 12 | u_flux_std | 1759.16 | Filter u stats |
| 13 | color_u_r_max | 1537.83 | Color |
| 14 | r_stetson_j | 1503.44 | Variability (filter r) |
| 15 | g_flux_std | 1448.22 | Filter g stats |
| 16 | i_flux_max | 1382.25 | Filter i stats |
| 17 | color_u_g_mean | 1337.81 | Color |
| 18 | g_decline_powerlaw_r2 | 1311.63 | TDE physics |
| 19 | peak_coherence | 1282.68 | Multi-band |
| 20 | z_flux_max | 1125.73 | Filter z stats |

**Insights**:
- **Variability features** chiếm ưu thế (i_excess_var, z_stetson_j, g_excess_var)
- **TDE physics features** rất quan trọng (power-law R², parabolic fits)
- **Filter i và g** đóng vai trò chủ đạo
- **Multi-band features** (colors, peak coherence) có tác động đáng kể

#### Kết quả cuối cùng:   F1 Score: 0.5199 và Optimal Threshold: 0.300, Public leaderboard: 0.6035
## 6. Kết luận
✅ **Xây dựng thành công pipeline hoàn chỉnh** cho bài toán phân loại TDE:
- Feature engineering with physics knowledge
- Advanced ensemble learning
- Optimization for imbalanced data

✅ **Ensemble LightGBM + XGBoost**:
- Stability từ LightGBM
- Diversity từ XGBoost
- Weighted combination tối ưu

✅ **Feature importance insights**:
- Variability features 
- TDE physics (power-law) có hiệu quả cao
- Multi-band information (colors, peak coherence) hiệu quả

