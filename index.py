import os
import sys
import socketserver

# Fix cho lỗi UnixStreamServer trên Windows
if not hasattr(socketserver, "UnixStreamServer"):
   socketserver.UnixStreamServer = socketserver.TCPServer
    
# Trỏ đúng đến thư mục bạn vừa cài/tải
os.environ['JAVA_HOME'] = r"D:\software\java11" # Đường dẫn cài Java
os.environ['HADOOP_HOME'] = r"D:\software\spark" # Thư mục chứa thư mục bin
os.environ['PATH'] = os.environ['PATH'] + r";D:\software\spark\bin" # Thêm đường dẫn hadoop bin vào PATH

# Chỉ định Python cho Spark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml import PipelineModel
import pandas as pd
import numpy as np

# 1. Khởi tạo Spark

from pyspark.sql.types import DoubleType

def process_national_data(file_path):
   # 1. Đọc dữ liệu thô
   df = spark.read.csv(file_path, header=True, inferSchema=True)
   
   # 2. Xử lý cột Date
   df = df.withColumn("Date", F.to_date(F.col("Date")))
   
   # 3. Danh sách TOÀN BỘ các cột số cần ép kiểu (loại bỏ Location, Risk_MM, v.v.)
   # Đây là các chỉ số khí tượng sẽ được tính trung bình toàn quốc
   numeric_cols = [
      "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
      "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
      "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
      "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
   ]
   
   # 4. Ép kiểu hàng loạt sang Double (Khắc phục lỗi toàn String)
   for col_name in numeric_cols:
      # Kiểm tra xem cột có trong file không để tránh lỗi
      if col_name in df.columns:
         df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
   return df
 
def display_schema_and_stats(df):
   """
   Hàm hiển thị thông tin để chụp ảnh báo cáo
   """
   print("\n" + "="*20 + " KẾT QUẢ KIỂM TRA DỮ LIỆU " + "="*20)
   
   # [HABD: Screenshot 1] Cấu trúc Schema
   print("\n--- 1. Cấu trúc Schema (df.printSchema) ---")
   df.printSchema()
   
   # [HABD: Screenshot 2] Kích thước dữ liệu
   print("\n--- 2. Kích thước dữ liệu (df.count) ---")
   total_rows = df.count()
   total_cols = len(df.columns)
   print(f"Tổng số dòng: {total_rows} (Bản ghi)")
   print(f"Tổng số cột: {total_cols} (Thuộc tính)")
   
   # [HABD: Screenshot 3] Thống kê mô tả
   print("\n--- 3. Thống kê mô tả (df.describe) ---")
   # Chọn cột tiêu biểu để bảng đẹp, dễ chụp
   df.describe(['MinTemp', 'MaxTemp', 'Rainfall']).show()
   print("="*65 + "\n")
   
   # --- HÀM TẠO DATA DICTIONARY (CẬP NHẬT) ---
def get_national_data_dictionary(df, csv_filename=None):
   """
   Tạo bảng từ điển dữ liệu dựa trên Schema thực tế của bảng Quốc gia
   """
   # Mô tả ý nghĩa các cột (Sau khi đã tính trung bình)
   descriptions = {
      "Date": "Ngày quan trắc",
      "MinTemp": "Nhiệt độ thấp nhất trung bình cả nước (°C)",
      "MaxTemp": "Nhiệt độ cao nhất trung bình cả nước (°C)",
      "Rainfall": "Lượng mưa trung bình cả nước (mm)",
      "Evaporation": "Lượng bốc hơi trung bình (mm)",
      "Sunshine": "Số giờ nắng trung bình (giờ)",
      "WindGustSpeed": "Tốc độ gió giật trung bình (km/h)",
      "WindDir9am": "Hướng gió thổi lúc 9h sáng",
      "WindDir3pm": "Hướng gió thổi lúc 3h chiều",
      "Humidity9am": "Độ ẩm trung bình lúc 9h sáng (%)",
      "Humidity3pm": "Độ ẩm trung bình lúc 3h chiều (%)",
      "Pressure9am": "Áp suất trung bình lúc 9h sáng (hPa)",
      "Pressure3pm": "Áp suất trung bình lúc 3h chiều (hPa)",
      "Cloud9am": "Lượng mây trung bình lúc 9h sáng (oktas)",
      "Cloud3pm": "Lượng mây trung bình lúc 3h chiều (oktas)",
      "Temp9am": "Nhiệt độ trung bình lúc 9h sáng (°C)",
      "Temp3pm": "Nhiệt độ trung bình lúc 3h chiều (°C)",
      "WindSpeed9am": "Tốc độ gió trung bình 9h sáng",
      "WindSpeed3pm": "Tốc độ gió trung bình 3h chiều",
      "RainToday": "Hôm nay có mưa không? (Yes/No)",
      "RainTomorrow": "Ngày mai có mưa không? (Biến mục tiêu)"
   }
   
   data = []
   for field in df.schema:
      name = field.name
      # Lấy kiểu dữ liệu thực tế từ Spark
      dtype = field.dataType.simpleString()
      desc = descriptions.get(name, "Chỉ số trung bình khác")
      data.append((name, dtype, desc))
   
   dict_df = pd.DataFrame(data, columns=["Tên thuộc tính", "Kiểu dữ liệu (Spark)", "Mô tả ý nghĩa"])
   
   print("\n" + "="*30 + " DATA DICTIONARY (DỮ LIỆU QUỐC GIA) " + "="*30)
   print(dict_df.to_string(index=False))
   print("="*90 + "\n")
   
   if csv_filename:
      try:
         # encoding='utf-8-sig' giúp Excel đọc tiếng Việt không bị lỗi
         dict_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
         print(f">>> Đã xuất bảng mô tả ra file: {csv_filename}")
      except Exception as e:
         print(f">>> Lỗi khi xuất file CSV: {e}")
         
   return dict_df
 
def get_advanced_statistics(df):
   """
   Hàm tính toán và hiển thị bảng thống kê chi tiết:
   Count, Mean, Stddev, Min, 25%, 50%, 75%, Max, Variance, IQR
   """
   target_cols = ["MinTemp", "MaxTemp", "Rainfall"]
   
   # Dictionary để lưu kết quả
   stats_data = {}
   
   print("Đang tính toán các chỉ số thống kê nâng cao...")
   
   for col in target_cols:
      # 1. Tính các chỉ số cơ bản & Phương sai (Variance)
      aggs = df.select(
         F.count(col).alias("count"),
         F.mean(col).alias("mean"),
         F.stddev(col).alias("stddev"),
         F.min(col).alias("min"),
         F.max(col).alias("max"),
         F.variance(col).alias("var") # Tính phương sai
      ).collect()[0]
      
      # 2. Tính phân vị (Percentiles) bằng approxQuantile (Chính xác cho dữ liệu lớn)
      # 0.25 = Q1, 0.5 = Median, 0.75 = Q3
      quantiles = df.approxQuantile(col, [0.25, 0.5, 0.75], 0.01)
      q1, median, q3 = quantiles[0], quantiles[1], quantiles[2]
      
      # 3. Tính IQR (Interquartile Range)
      iqr = q3 - q1
      
      # Lưu vào dict
      stats_data[col] = {
         "count": int(aggs["count"]),
         "mean": round(aggs["mean"], 2),
         "stddev": round(aggs["stddev"], 2),
         "var": round(aggs["var"], 2),      # Phương sai
         "min": round(aggs["min"], 2),
         "25% (Q1)": round(q1, 2),
         "50% (Median)": round(median, 2),
         "75% (Q3)": round(q3, 2),
         "max": round(aggs["max"], 2),
         "iqr": round(iqr, 2)               # Khoảng tứ phân vị
      }
      
   # Chuyển đổi sang Pandas DataFrame để hiển thị đẹp
   pdf_stats = pd.DataFrame(stats_data)
   
   # Sắp xếp lại thứ tự các dòng (Index) cho đúng logic báo cáo
   ordered_index = ["count", "mean", "stddev", "var", "min", "25% (Q1)", "50% (Median)", "75% (Q3)", "max", "iqr"]
   pdf_stats = pdf_stats.reindex(ordered_index)
   
   print("\n" + "="*20 + "BẢNG THỐNG KÊ MÔ TẢ CHI TIẾT " + "="*20)
   print(pdf_stats)
   print("="*70 + "\n")
   
   # Xuất ra CSV nếu cần
   pdf_stats.to_csv("advanced_statistics.csv", encoding='utf-8-sig')
   return pdf_stats
 
 
def load_and_clean_data(file_path):
   # 1. Đọc dữ liệu
   df = spark.read.csv(file_path, header=True, inferSchema=True)
   
   # 2. Ép kiểu dữ liệu Date và các cột số (Quan trọng)
   df = df.withColumn("Date", F.to_date(F.col("Date")))
   
   # Danh sách các cột cần đưa về kiểu số (Double)
   numeric_cols = ["MinTemp", "MaxTemp", "Rainfall"]
   
   for col_name in numeric_cols:
      # Ép kiểu từ String sang Double để Imputer có thể hoạt động
      df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
   print("--- Schema sau khi ép kiểu dữ liệu ---")
   df.printSchema()
   
   # 3. Xử lý Null bằng Imputer
   # Bây giờ các cột đã là numeric, Imputer sẽ chạy mượt mà
   imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols).setStrategy("mean")
   df_clean = imputer.fit(df).transform(df)
   
   return df_clean

def plot_histograms(df):
   """
   Vẽ và lưu 3 biểu đồ Histogram phân phối dữ liệu
   """
   print(">>> Đang chuyển đổi dữ liệu sang Pandas để vẽ biểu đồ (Vui lòng đợi)...")
   
   # Chỉ lấy 3 cột cần thiết để tiết kiệm bộ nhớ khi toPandas()
   pdf = df.select("MinTemp", "MaxTemp", "Rainfall").toPandas()
   
   # Thiết lập khung hình (1 hàng, 3 cột)
   fig, axes = plt.subplots(1, 3, figsize=(18, 6))
   
   # 1. Biểu đồ MinTemp
   sns.histplot(pdf["MinTemp"], kde=True, ax=axes[0], color="skyblue", bins=30)
   axes[0].set_title("Phân phối MinTemp (°C)")
   axes[0].set_xlabel("Nhiệt độ")
   axes[0].set_ylabel("Tần suất")
   
   # 2. Biểu đồ MaxTemp
   sns.histplot(pdf["MaxTemp"], kde=True, ax=axes[1], color="salmon", bins=30)
   axes[1].set_title("Phân phối MaxTemp (°C)")
   axes[1].set_xlabel("Nhiệt độ")
   axes[1].set_ylabel("Tần suất")
   
   # 3. Biểu đồ Rainfall
   # Lưu ý: Rainfall thường lệch trái rất nhiều (nhiều ngày không mưa)
   # nên dùng log scale cho trục Y để dễ nhìn hơn
   sns.histplot(pdf["Rainfall"], kde=False, ax=axes[2], color="green", bins=30)
   axes[2].set_title("Phân phối Rainfall (mm)")
   axes[2].set_xlabel("Lượng mưa")
   axes[2].set_ylabel("Tần suất (Log Scale)")
   axes[2].set_yscale('log') # Dùng thang đo Logarit để thấy rõ các cột thấp
   
   # Căn chỉnh layout
   plt.tight_layout()
   
   # Lưu thành file ảnh để chèn vào báo cáo
   filename = "HABD_bieu_do_histogram.png"
   plt.savefig(filename, dpi=300) # dpi=300 để ảnh sắc nét khi in ấn
   print(f">>> Đã lưu biểu đồ thành công: {filename}")
   
   # Hiển thị lên màn hình
   plt.show()

# --- HÀM VẼ BIỂU ĐỒ BOXPLOT---
def plot_boxplots(df):
   """
   Vẽ và lưu 3 biểu đồ Boxplot để phát hiện điểm ngoại lai (Outliers)
   """
   print(">>> Đang tạo biểu đồ Boxplot (Box & Whisker)...")
   
   # Chuyển đổi dữ liệu sang Pandas
   pdf = df.select("MinTemp", "MaxTemp", "Rainfall").toPandas()
   
   # Thiết lập khung hình (1 hàng, 3 cột)
   fig, axes = plt.subplots(1, 3, figsize=(18, 6))
   
   # 1. Boxplot MinTemp
   sns.boxplot(y=pdf["MinTemp"], ax=axes[0], color="skyblue")
   axes[0].set_title("Boxplot MinTemp - Nhiệt độ thấp nhất")
   axes[0].set_ylabel("Nhiệt độ (°C)")
   
   # 2. Boxplot MaxTemp
   sns.boxplot(y=pdf["MaxTemp"], ax=axes[1], color="salmon")
   axes[1].set_title("Boxplot MaxTemp - Nhiệt độ cao nhất")
   axes[1].set_ylabel("Nhiệt độ (°C)")
   
   # 3. Boxplot Rainfall
   # Rainfall có outliers rất xa, nên giữ nguyên scale để thấy độ cực đoan
   sns.boxplot(y=pdf["Rainfall"], ax=axes[2], color="lightgreen")
   axes[2].set_title("Boxplot Rainfall - Lượng mưa")
   axes[2].set_ylabel("Lượng mưa (mm)")
   
   # Tinh chỉnh giao diện
   plt.tight_layout()
   
   # Lưu file ảnh
   filename = "HABD_bieu_do_boxplot_outliers.png"
   plt.savefig(filename, dpi=300)
   print(f">>> Đã lưu biểu đồ Boxplot: {filename}")
   
   plt.show()

def plot_heatmap(df):
   """[HABD: Biểu đồ Heatmap]"""
   pdf = df.select("MinTemp", "MaxTemp", "Rainfall").toPandas()
   plt.figure(figsize=(8, 6))
   sns.heatmap(pdf.corr(), annot=True, cmap='coolwarm', fmt=".2f")
   plt.title("Ma trận tương quan Pearson")
   plt.show()
   
# ---  HÀM MINH CHỨNG XỬ LÝ NULL (CẬP NHẬT: RAINFALL = 0) ---
def demonstrate_null_handling(df):
   """
   Nhiệt độ -> Mean, Mưa -> 0
   """
   # Các cột cần theo dõi
   target_cols = ["MinTemp", "MaxTemp", "Rainfall"]
   
   # =========================================================================
   # THỐNG KÊ NULL TRƯỚC KHI XỬ LÝ
   # =========================================================================
   print("\n" + "="*20 + " SỐ LƯỢNG NULL TRƯỚC XỬ LÝ " + "="*20)
   print(">>> Đang kiểm tra dữ liệu thô...")
   df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in target_cols]).show()
   
   # =========================================================================
   # [HÌNH 2] ĐOẠN CODE XỬ LÝ (CHỤP PHẦN NÀY)
   # =========================================================================
   print("\n" + "="*20 + " CODE XỬ LÝ (NHIỆT ĐỘ=MEAN, MƯA=0) " + "="*20)
   print(">>> Đang chạy xử lý: Min/MaxTemp dùng Mean, Rainfall điền 0...")
   
   # 1. Xử lý Nhiệt độ (MinTemp, MaxTemp) bằng Trung bình (Mean)
   imputer = Imputer(
      inputCols=["MinTemp", "MaxTemp"], 
      outputCols=["MinTemp", "MaxTemp"]
   ).setStrategy("mean")
   
   df_temp_clean = imputer.fit(df).transform(df)
   
   # 2. Xử lý Lượng mưa (Rainfall) bằng cách điền 0 (fillna)
   # Lý do: Null thường có nghĩa là hôm đó không mưa
   df_clean = df_temp_clean.fillna(0, subset=["Rainfall"])
   
   print(">>> Đã xử lý xong!")

   # =========================================================================
   # THỐNG KÊ NULL SAU KHI XỬ LÝ
   # =========================================================================
   print("\n" + "="*20 + " [HÌNH 3] SỐ LƯỢNG NULL SAU XỬ LÝ " + "="*20)
   print(">>> Kiểm tra lại để chứng minh sạch 100%...")
   
   df_clean.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in target_cols]).show()
   print("="*70 + "\n")
   
   return df_clean
   

def visualize_scaling_effect(df_spark, feature_cols):
   # Lấy mẫu khoảng 100 dòng để vẽ biểu đồ
   pdf = df_spark.select(feature_cols + ["features"]).limit(100).toPandas()
   
   # Chọn đại diện 2 đặc trưng có biên độ khác nhau để so sánh
   # Ví dụ: MinTemp (tầm 20-30) và Rainfall (tầm 0-100 hoặc hơn)
   col1 = feature_cols[0] # MinTemp_Lag1
   col2 = feature_cols[2] # Rainfall_Lag1 (Thường mưa có biên độ rộng và outliers lớn)
   
   # Tách dữ liệu sau khi scale từ cột Vector ra để vẽ
   # Spark lưu dưới dạng DenseVector, cần chuyển về numpy array
   scaled_data = np.array(pdf["features"].tolist())
   
   # --- VẼ BIỂU ĐỒ ---
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # 1. TRƯỚC KHI CHUẨN HÓA (Raw Data)
   axes[0].boxplot([pdf[col1], pdf[col2]], labels=[col1, col2])
   axes[0].set_title("TRƯỚC khi chuẩn hóa (Original Scales)")
   axes[0].set_ylabel("Giá trị thực tế")
   axes[0].grid(True, linestyle='--', alpha=0.6)
      
   # 2. SAU KHI CHUẨN HÓA (Scaled Data)
   # Lấy index tương ứng trong vector
   idx1 = 0 
   idx2 = 2
   axes[1].boxplot([scaled_data[:, idx1], scaled_data[:, idx2]], labels=[col1, col2])
   axes[1].set_title("SAU khi chuẩn hóa (StandardScaler)")
   axes[1].set_ylabel("Z-Score (Độ lệch chuẩn)")
   axes[1].grid(True, linestyle='--', alpha=0.6)
 
   
   plt.tight_layout()
   plt.show()



def create_multi_output_features(df):
   # drop cols
   """Tạo Lag 5 ngày (Features) và Lead 2 ngày cho cả 3 biến (Labels)"""
   df = df.select("Date","MinTemp", "MaxTemp", "Rainfall")
   
   windowSpec = Window.orderBy("Date")
   
   # 1. Tạo Đặc trưng Lag (5 ngày qua cho cả 3 biến)
   for i in range(1, 6):
      df = df.withColumn(f"MinTemp_L{i}", F.lag("MinTemp", i).over(windowSpec))
      df = df.withColumn(f"MaxTemp_L{i}", F.lag("MaxTemp", i).over(windowSpec))
      df = df.withColumn(f"Rainfall_L{i}", F.lag("Rainfall", i).over(windowSpec))
   
   # 2. Tạo Nhãn Lead (Dự báo 2 ngày tới cho cả 3 biến)
   targets = []
   for d in [1, 2]: # Ngày 1 và Ngày 2
      for var in ["MinTemp", "MaxTemp", "Rainfall"]:
         col_name = f"Target_{var}_D{d}"
         df = df.withColumn(col_name, F.lead(var, d).over(windowSpec))
         targets.append(col_name)
   
   df_ml = df.dropna()
   print(f"Tổng số đặc trưng đầu vào: 15")
   print(f"Tổng số nhãn cần dự báo: {len(targets)}")
   
   return df_ml, targets

def vectorize_and_scale(df, feature_cols):
   """
   Input: DataFrame sạch và danh sách tên các cột Features (Lag)
   Output: DataFrame đã có cột 'features' chuẩn hóa
   """
   print("--- BẮT ĐẦU VECTOR HÓA VÀ CHUẨN HÓA ---")
   
   # 1. Vector Assembler: Gom tất cả cột đặc trưng vào 1 vector
   assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
   df_vector = assembler.transform(df)
   
   # 2. Standard Scaler: Chuẩn hóa dữ liệu
   # withMean=True, withStd=True: Đưa dữ liệu về phân phối chuẩn (Mean=0, Std=1)
   scaler = StandardScaler(inputCol="features_raw", outputCol="features", 
                           withStd=True, withMean=True)
   
   # Fit mô hình scaler để tính toán Mean/Std của toàn bộ dữ liệu
   scaler_model = scaler.fit(df_vector)
   df_scaled = scaler_model.transform(df_vector)
   
   print("Đã xong. Cột kết quả cuối cùng dùng để train model là: 'features'")
   return df_scaled

def split_train_test_chronological(df, train_ratio=0.8):
   """
   Chia dữ liệu theo thứ tự thời gian.
   - df: DataFrame đầu vào (đã chuẩn hóa).
   - train_ratio: Tỷ lệ tập train (mặc định 0.8).
   """
   # 1. Đếm tổng số dòng
   total_rows = df.count()
   train_count = int(total_rows * train_ratio)
   
   # 2. Đánh số thứ tự dòng theo ngày (tăng dần)
   # Window này đảm bảo dòng 1 là ngày cũ nhất
   windowSpec = Window.orderBy("Date")
   df_indexed = df.withColumn("index", F.row_number().over(windowSpec))
   
   # 3. Cắt dữ liệu dựa trên số thứ tự
   train_set = df_indexed.filter(F.col("index") <= train_count).drop("index")
   
   # Test
   test_set = df_indexed.filter(F.col("index") > train_count).drop("index")
   
   # 4. In kết quả kiểm tra
   print(f"--- KẾT QUẢ CHIA TRAIN/TEST ({int(train_ratio*100)}/{int((1-train_ratio)*100)}) ---")
   print(f"Tổng số dòng: {total_rows}")
   print(f"Train set (Huấn luyện): {train_set.count()} dòng")
   print(f"Test set (Kiểm thử)   : {test_set.count()} dòng")
   
   return train_set, test_set

def visualize_temperature_trend(df):
   """
   Vẽ biểu đồ đường thể hiện nhiệt độ trung bình theo thời gian.
   Input: DataFrame chứa cột Date, MinTemp, MaxTemp
   """
   print("--- ĐANG XỬ LÝ DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ ---")
   
   # 1. Tính nhiệt độ trung bình (MeanTemp) = (Min + Max) / 2
   # Sau đó GroupBy Date để tính trung bình toàn quốc (nếu dữ liệu có nhiều trạm đo/ngày)
   df_trend = df.withColumn("Temp_Mean", (F.col("MinTemp") + F.col("MaxTemp")) / 2) \
               .groupBy("Date") \
               .agg(F.avg("Temp_Mean").alias("Aggregated_Mean_Temp")) \
               .orderBy("Date")
   
   # 2. Chuyển sang Pandas để vẽ (Dữ liệu sau khi group by theo ngày sẽ nhỏ, convert an toàn)
   pdf = df_trend.toPandas()
   
   # Đảm bảo cột Date là kiểu datetime của Pandas
   pdf['Date'] = pd.to_datetime(pdf['Date'])

   # 3. Vẽ biểu đồ với Matplotlib
   plt.figure(figsize=(14, 6)) # Kích thước ảnh rộng cho dễ nhìn
   
   # Vẽ đường line
   plt.plot(pdf['Date'], pdf['Aggregated_Mean_Temp'], 
            color='#1f77b4', linewidth=2, label='Nhiệt độ TB Toàn quốc')
   
   # Trang trí biểu đồ
   plt.title('Biến động Nhiệt độ Trung bình Theo Thời gian', fontsize=16, fontweight='bold')
   plt.xlabel('Thời gian (Năm/Tháng)', fontsize=12)
   plt.ylabel('Nhiệt độ (°C)', fontsize=12)
   plt.grid(True, linestyle='--', alpha=0.6) # Thêm lưới mờ cho dễ tra cứu
   plt.legend()
   
   # Xoay nhãn ngày tháng cho đỡ bị đè lên nhau
   plt.xticks(rotation=45)
   
   plt.tight_layout()
   plt.show()
 
def train_save_final_models(train_df, feature_cols, targets, save_dir="saved_models"):
   """
   Train 1 lần duy nhất cho 6 target (cả LR và RF) và lưu model.
   """
   if not os.path.exists(save_dir):
      os.makedirs(save_dir)

   print(f"--- BẮT ĐẦU HUẤN LUYỆN (Train 1 lần) ---")
   
   # Định nghĩa Assembler & Scaler chung
   assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
   scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

   for target in targets:
      print(f"--> Processing Target: {target}")
      
      # --- 1. TRAIN & SAVE LINEAR REGRESSION ---
      lr = LinearRegression(featuresCol="features", labelCol=target, regParam=0.1)
      pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
      model_lr = pipeline_lr.fit(train_df)
      
      # Lưu: saved_models/LR_TargetName
      path_lr = f"{save_dir}/LR_{target}"
      model_lr.write().overwrite().save(path_lr)
      
      # --- 2. TRAIN & SAVE RANDOM FOREST ---
      rf = RandomForestRegressor(featuresCol="features", labelCol=target, numTrees=100, seed=42)
      pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
      model_rf = pipeline_rf.fit(train_df)
      
      # Lưu: saved_models/RF_TargetName
      path_rf = f"{save_dir}/RF_{target}"
      model_rf.write().overwrite().save(path_rf)
      
   print(f"[DONE] Đã lưu toàn bộ model vào thư mục '{save_dir}'")
   

def evaluate_bootstrap_10_runs(test_df, targets, model_dir="saved_models", output_excel="bootstrap_evaluation.xlsx"):
   print("--- BẮT ĐẦU ĐÁNH GIÁ 10 LẦN (BOOTSTRAP TEST SET) ---")
   
   results = []
   
   # 1. LOAD TOÀN BỘ MODEL LÊN BỘ NHỚ TRƯỚC (Để vòng lặp chạy nhanh)
   loaded_models = {}
   print("Đang load models từ ổ cứng...")
   for target in targets:
      loaded_models[f"LR_{target}"] = PipelineModel.load(f"{model_dir}/LR_{target}")
      loaded_models[f"RF_{target}"] = PipelineModel.load(f"{model_dir}/RF_{target}")

   # 2. VÒNG LẶP 10 LẦN ĐÁNH GIÁ
   for i in range(1, 11):
      # Tạo seed khác nhau cho mỗi lần lấy mẫu
      seed = 42 + i
      
      bootstrap_test = test_df.sample(withReplacement=True, fraction=1.0, seed=seed)
      
      print(f"Run {i}/10: Evaluating on Bootstrap Sample (Seed={seed})...")
      
      row_metrics = {"Run_ID": i}
      
      # Biến tạm tính trung bình
      sums = {"LR_D1": 0, "LR_D2": 0, "RF_D1": 0, "RF_D2": 0}
      
      for target in targets:
         is_day_1 = "Next1" in target or "D1" in target
         
         # --- Đánh giá LR ---
         model_lr = loaded_models[f"LR_{target}"]
         preds_lr = model_lr.transform(bootstrap_test)
         evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
         rmse_lr = evaluator.evaluate(preds_lr)
         
         # --- Đánh giá RF ---
         model_rf = loaded_models[f"RF_{target}"]
         preds_rf = model_rf.transform(bootstrap_test)
         rmse_rf = evaluator.evaluate(preds_rf)
         
         # Lưu chi tiết
         row_metrics[f"RMSE_LR_{target}"] = rmse_lr
         row_metrics[f"RMSE_RF_{target}"] = rmse_rf
         
         # Cộng dồn
         if is_day_1:
               sums["LR_D1"] += rmse_lr
               sums["RF_D1"] += rmse_rf
         else:
               sums["LR_D2"] += rmse_lr
               sums["RF_D2"] += rmse_rf
      
      # Tính trung bình (chia 3 vì có Min, Max, Rain)
      row_metrics["LR_Avg_Day1"] = sums["LR_D1"] / 3
      row_metrics["LR_Avg_Day2"] = sums["LR_D2"] / 3
      row_metrics["RF_Avg_Day1"] = sums["RF_D1"] / 3
      row_metrics["RF_Avg_Day2"] = sums["RF_D2"] / 3
      
      results.append(row_metrics)

   # 3. XUẤT EXCEL
   df_results = pd.DataFrame(results)
   
   # Thêm dòng trung bình tổng
   avg_row = df_results.mean(numeric_only=True)
   avg_row["Run_ID"] = "FINAL_AVERAGE"
   
   df_final = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
   
   # Sắp xếp cột: Đưa các cột Avg ra trước cho dễ nhìn
   cols_order = ['Run_ID', 'LR_Avg_Day1', 'RF_Avg_Day1', 'LR_Avg_Day2', 'RF_Avg_Day2']
   remaining = [c for c in df_final.columns if c not in cols_order]
   df_final = df_final[cols_order + remaining]
   
   df_final.to_excel(output_excel, index=False)
   print(f"\n[HOÀN TẤT] File báo cáo chi tiết: {output_excel}")
   return df_final


def plot_zoom_index_based(train_df, test_df, target_col, 
                        model_type="rf", model_dir="saved_models", 
                        window_size=100):
   # 1. Load Model
   model_name = f"{model_type.upper()}_{target_col}"
   try:
      loaded_model = PipelineModel.load(f"{model_dir}/{model_name}")
   except:
      print(f"Lỗi: Không tìm thấy model {model_name}")
      return

   print(f"--- Đang xử lý dữ liệu (Vẽ theo Index) ---")

   # 2. Lấy dữ liệu (Train Tail + Test Head)
   train_tail = train_df.orderBy(F.col("Date").desc()).limit(window_size).withColumn("Type", F.lit("Train"))
   test_head = test_df.orderBy(F.col("Date").asc()).limit(window_size).withColumn("Type", F.lit("Test"))
   
   # Gộp và sort lại theo ngày để có thứ tự đúng
   cols = train_tail.columns
   combined_df = train_tail.select(cols).union(test_head.select(cols))
   
   # 3. Dự báo
   predictions = loaded_model.transform(combined_df)
   
   # 4. Chuyển sang Pandas
   pdf = predictions.select("Date", target_col, "prediction", "Type") \
                  .orderBy("Date") \
                  .toPandas()

   pdf = pdf.reset_index(drop=True)
   
   # Tìm vị trí chuyển giao (điểm bắt đầu của Test trong index mới)
   # Lấy index của dòng đầu tiên có Type='Test'
   split_idx = pdf[pdf['Type'] == 'Test'].index.min()
   
   # Format cột Date thành chuỗi ngắn gọn để hiển thị trục hoành
   pdf['Date_Str'] = pd.to_datetime(pdf['Date']).dt.strftime('%d/%m/%Y')

   # 5. Vẽ Biểu Đồ
   plt.figure(figsize=(14, 6))
   
   # --- VẼ NỀN ---
   # Vùng Train (Từ 0 đến split_idx)
   plt.axvspan(0, split_idx, color='#e0e0e0', alpha=0.5, label='Train (Quá khứ)')
   # Vùng Test (Từ split_idx đến hết)
   plt.axvspan(split_idx, len(pdf)-1, color='white', alpha=0.5, label='Test (Tương lai)')
   
   # --- VẼ DỮ LIỆU (Dùng pdf.index làm trục X) ---
   plt.plot(pdf.index, pdf[target_col], 
            color='#1f77b4', label='Thực tế', linewidth=2)
   
   plt.plot(pdf.index, pdf['prediction'], 
            color='#d62728', label='Dự báo', 
            linestyle='--', linewidth=2)
   
   xticks_locs = np.linspace(0, len(pdf)-1, 10, dtype=int)
   xticks_labels = pdf['Date_Str'].iloc[xticks_locs]
   
   plt.xticks(xticks_locs, xticks_labels, rotation=45)
   
   # Vạch ngăn cách
   plt.axvline(x=split_idx, color='black', linewidth=2)
   
   algo = "Random Forest" if model_type == 'rf' else "Linear Reg"
   plt.title(f'Biểu đồ dự báo: {target_col} - {algo}', fontsize=14, fontweight='bold')
   plt.xlabel('Thời gian (Theo tháng)', fontsize=12)
   plt.ylabel('Giá trị', fontsize=12)
   plt.legend()
   plt.grid(True, linestyle=':', alpha=0.6)
   plt.tight_layout()
   plt.show()


if __name__ == "__main__":
   try:
      spark = SparkSession.builder \
         .appName("WeatherProject") \
         .master("local[*]") \
         .config("spark.driver.host", "127.0.0.1") \
         .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
         .getOrCreate()
      print("Khởi tạo Spark thành công! Bạn có thể bắt đầu làm bài tập lớn.")
   except Exception as e:
      print(f"Vẫn còn lỗi: {e}")
      
      
   df_national = process_national_data("weatherAUS.csv")
   
   # display_schema_and_stats(df_national)
   
   # get_advanced_statistics(df_national)
   
   # plot_histograms(df_national)
   # plot_boxplots(df_national)
   
   df_clean = demonstrate_null_handling(df_national)
   df_ml, target_columns = create_multi_output_features(df_clean)
   # show five rows of the first few columns
   # df_ml.select(df_ml.columns[:18] + target_columns[:3]).show(5)
   # df_ml.show(3, truncate=False)
   
   feature_cols = [
      "MinTemp_L1", "MaxTemp_L1", "Rainfall_L1",
      "MinTemp_L2", "MaxTemp_L2", "Rainfall_L2",
      "MinTemp_L3", "MaxTemp_L3", "Rainfall_L3",
      "MinTemp_L4", "MaxTemp_L4", "Rainfall_L4",
      "MinTemp", "MaxTemp", "Rainfall"
      ]
   
   # df_ready = vectorize_and_scale(df_ml, feature_cols)
   # visualize_scaling_effect(df_ready, feature_cols)
   
   train_set, test_set = split_train_test_chronological(df_ml, train_ratio=0.8)
   # Train và đánh giá mô hình cho Target_MinTemp_D1,
   
   targets = [
      "Target_MinTemp_D1", "Target_MaxTemp_D1", "Target_Rainfall_D1",
      "Target_MinTemp_D2", "Target_MaxTemp_D2", "Target_Rainfall_D2"
   ]
   
   # train_save_final_models(train_set, feature_cols, targets, save_dir="saved_models")
   
   # report = evaluate_bootstrap_10_runs(test_set, targets, model_dir="saved_models", output_excel="bootstrap_evaluation.xlsx")
   # visualize_temperature_trend(df_clean)
   # plot_heatmap(df_clean)

   plot_zoom_index_based(train_set, test_set, targets[2], model_type="rf", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[0], model_type="lr", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[1], model_type="lr", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[2], model_type="lr", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[3], model_type="lr", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[4], model_type="lr", window_size=100)
   plot_zoom_index_based(train_set, test_set, targets[5], model_type="lr", window_size=100)
   