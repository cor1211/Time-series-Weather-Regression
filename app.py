import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

# --- CẤU HÌNH TRANG ---
st.set_page_config(
   page_title="Weather Forecast Demo - Big Data Project",
   page_icon="🌤️",
   layout="wide"
)

# --- 1. KHỞI TẠO SPARK & CACHE (Để không phải load lại nhiều lần) ---
import socketserver
import sys

@st.cache_resource
def get_spark_session():
   if not hasattr(socketserver, "UnixStreamServer"):
      socketserver.UnixStreamServer = socketserver.TCPServer
    
   # Trỏ đúng đến thư mục vừa cài/tải
   os.environ['JAVA_HOME'] = r"D:\software\java11" # Đường dẫn cài Java
   os.environ['HADOOP_HOME'] = r"D:\software\spark" # Thư mục chứa thư mục bin
   os.environ['PATH'] = os.environ['PATH'] + r";D:\software\spark\bin" # Thêm đường dẫn hadoop bin vào PATH

   # Chỉ định Python cho Spark
   os.environ['PYSPARK_PYTHON'] = sys.executable
   os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
   
   try:
      spark = SparkSession.builder \
         .appName("WeatherProject") \
         .master("local[*]") \
         .config("spark.driver.host", "127.0.0.1") \
         .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
         .getOrCreate()
      print("Khởi tạo Spark thành công!")
      return spark
   except Exception as e:
      print(f"Vẫn còn lỗi: {e}")
   
   

@st.cache_resource
def load_models(model_dir="saved_models"):
   """
   Load toàn bộ 12 models (6 cho LR, 6 cho RF) vào bộ nhớ đệm
   """
   models = {}
   targets = [
      "Target_MinTemp_D1", "Target_MaxTemp_D1", "Target_Rainfall_D1",
      "Target_MinTemp_D2", "Target_MaxTemp_D2", "Target_Rainfall_D2"
   ]
   
   # Kiểm tra đường dẫn
   if not os.path.exists(model_dir):
      return None, "Không tìm thấy thư mục saved_models!"

   for algo in ["LR", "RF"]:
      for target in targets:
         path = f"{model_dir}/{algo}_{target}"
         try:
               # Load PipelineModel (bao gồm cả Scaler/Assembler)
               models[f"{algo}_{target}"] = PipelineModel.load(path)
         except Exception as e:
               print(f"Warning: Không load được {path}")
   return models, targets

# @st.cache_data
def load_test_data(data_path="test_data.parquet"):
   """
   Load dữ liệu Test. Cache data chuyển sang Pandas để UI chạy nhanh.
   """
   spark = get_spark_session()
   if os.path.exists(data_path):
      # Đọc Parquet (nhanh hơn CSV)
      df = spark.read.parquet(data_path)
   elif os.path.exists(data_path.replace("parquet", "csv")):
      # Fallback sang CSV nếu không có parquet
      df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path.replace("parquet", "csv"))
   else:
      return None
      
   # Sắp xếp theo ngày
   return df.orderBy("Date")

# --- 2. GIAO DIỆN CHÍNH ---

def main():
   st.title("🌤️ Hệ Thống Dự Báo Thời Tiết - Demo Spark MLlib")
   st.markdown("---")

   # Load dữ liệu
   spark = get_spark_session()
   
   # --- SIDEBAR: CẤU HÌNH ---
   st.sidebar.header("⚙️ Cấu hình dự báo")
   
   # 1. Load Data
   # Nếu đang chạy trên máy local mà chưa lưu file
   # Tốt nhất: lưu test_df.write.parquet("test_data.parquet") rồi load lại ở đây.
   test_spark_df = load_test_data("test_data.parquet") 
   
   if test_spark_df is None:
      st.error("⚠️ Không tìm thấy file dữ liệu test! Hãy lưu 'test_df' ra file 'test_data.parquet' hoặc '.csv'.")
      return

   # Chuyển một phần nhỏ sang Pandas để làm danh sách chọn ngày cho nhanh
   # Chỉ lấy cột Date và Index để tạo Dropdown
   date_options = test_spark_df.select("Date") \
                            .dropDuplicates(["Date"]) \
                            .orderBy("Date", ascending=False) \
                            .limit(50) \
                            .toPandas()
   date_options['Date'] = pd.to_datetime(date_options['Date'])
   
   selected_date = st.sidebar.selectbox(
      "Chọn Ngày Dự Báo (Ngày T):",
      options=date_options['Date'],
      format_func=lambda x: x.strftime('%d/%m/%Y')
   )

   # 2. Chọn Model
   model_type = st.sidebar.radio("Chọn Thuật Toán:", ["Random Forest (RF)", "Linear Regression (LR)"])
   algo_prefix = "RF" if "Random" in model_type else "LR"

   # 3. Load Models
   models, targets = load_models()
   if models is None:
      st.error(targets) # In lỗi đường dẫn
      return

   # --- MAIN COLUMN: XỬ LÝ DỰ BÁO ---
   
   # Lấy đúng dòng dữ liệu của ngày được chọn
   # Filter trên Spark DataFrame
   input_row_spark = test_spark_df.filter(F.col("Date") == selected_date)
   
   # Chuyển sang Pandas để hiển thị UI
   input_row_pdf = input_row_spark.toPandas()

   if input_row_pdf.empty:
      st.warning("Không tìm thấy dữ liệu cho ngày này.")
      return

   # --- PHẦN 1: THÔNG TIN ĐẦU VÀO (CONTEXT) ---
   st.subheader(f"📅 Thông tin đầu vào: {selected_date.strftime('%d/%m/%Y')}")
   
   # Hiển thị các chỉ số quá khứ (Lag 1) để người xem nắm bối cảnh
   # Giả sử tên cột là MinTemp_Lag1, MaxTemp_Lag1... 
   cols = st.columns(3)
   try:
      cols[0].metric("MinTemp (Hôm qua)", f"{input_row_pdf['MinTemp_L1'].iloc[0]} °C")
      cols[1].metric("MaxTemp (Hôm qua)", f"{input_row_pdf['MaxTemp_L1'].iloc[0]} °C")
      cols[2].metric("Rainfall (Hôm qua)", f"{input_row_pdf['Rainfall_L1'].iloc[0]} mm")
   except KeyError:
      st.info("Hiển thị cột Lag: Kiểm tra lại tên cột trong DataFrame (VD: MinTemp_Lag1 hay MinTemp_L1)")

   st.markdown("---")

   # --- PHẦN 2: THỰC HIỆN DỰ BÁO ---
   st.subheader(f"🚀 Kết quả Dự Báo ({model_type})")

   # Tạo 2 tab cho Ngày 1 và Ngày 2
   tab1, tab2 = st.tabs(["Dự Báo Ngày Mai (D1)", "Dự Báo Ngày Kia (D2)"])

   # Hàm helper để lấy kết quả
   def get_prediction(target_name, row_spark):
      model_key = f"{algo_prefix}_{target_name}"
      if model_key not in models:
         return 0.0, 0.0 # Model chưa train hoặc lỗi tên
         
      # Predict
      pred_df = models[model_key].transform(row_spark)
      pred_val = pred_df.select("prediction").collect()[0][0]
      
      # Lấy giá trị thực tế (Label) có sẵn trong test set để so sánh
      actual_val = pred_df.select(target_name).collect()[0][0]
      
      return pred_val, actual_val

   # --- TAB 1: NGÀY MAI (D1) ---
   with tab1:
      c1, c2, c3 = st.columns(3)
      
      # 1. MinTemp D1
      pred, actual = get_prediction("Target_MinTemp_D1", input_row_spark)
      delta = pred - actual
      c1.metric(label="Nhiệt độ Thấp nhất", value=f"{pred:.1f} °C", 
               delta=f"Lệch: {delta:.1f} °C", delta_color="inverse")
      c1.caption(f"Thực tế: {actual} °C")

      # 2. MaxTemp D1
      pred, actual = get_prediction("Target_MaxTemp_D1", input_row_spark)
      delta = pred - actual
      c2.metric(label="Nhiệt độ Cao nhất", value=f"{pred:.1f} °C", 
               delta=f"Lệch: {delta:.1f} °C", delta_color="inverse")
      c2.caption(f"Thực tế: {actual} °C")

      # 3. Rainfall D1
      pred, actual = get_prediction("Target_Rainfall_D1", input_row_spark)
      delta = pred - actual
      c3.metric(label="Lượng Mưa", value=f"{pred:.1f} mm", 
               delta=f"Lệch: {delta:.1f} mm", delta_color="inverse")
      c3.caption(f"Thực tế: {actual} mm")

      # Actionable Insight
      if pred > 5.0:
         st.warning("⚠️ Dự báo có mưa đáng kể! Nên mang theo ô/áo mưa.")
      elif pred > 0.5:
            st.info("ℹ️ Có khả năng mưa nhỏ.")
      else:
         st.success("☀️ Trời tạnh ráo.")

   # --- TAB 2: NGÀY KIA (D2) ---
   with tab2:
      c1, c2, c3 = st.columns(3)
      
      # 1. MinTemp D2
      pred, actual = get_prediction("Target_MinTemp_D2", input_row_spark)
      delta = pred - actual
      c1.metric(label="Nhiệt độ Thấp nhất", value=f"{pred:.1f} °C", delta=f"{delta:.1f}", delta_color="inverse")

      # 2. MaxTemp D2
      pred, actual = get_prediction("Target_MaxTemp_D2", input_row_spark)
      delta = pred - actual
      c2.metric(label="Nhiệt độ Cao nhất", value=f"{pred:.1f} °C", delta=f"{delta:.1f}", delta_color="inverse")

      # 3. Rainfall D2
      pred, actual = get_prediction("Target_Rainfall_D2", input_row_spark)
      delta = pred - actual
      c3.metric(label="Lượng Mưa", value=f"{pred:.1f} mm", delta=f"{delta:.1f}", delta_color="inverse")

   st.markdown("---")
   
   # --- PHẦN 3: BIỂU ĐỒ PHÂN TÍCH ---
   
   # Lấy 100 ngày xung quanh ngày được chọn để vẽ biểu đồ
   # Logic: Filter ngày > selected_date - 50 và ngày < selected_date + 50
   # Để đơn giản cho demo, ta vẽ 100 ngày *sau* ngày được chọn
   
   st.subheader("📈 Phân tích xu hướng: Nhiệt độ & Lượng mưa (30 ngày tới)")
   
   # 1. Chuẩn bị dữ liệu vẽ
   start_plot_date = F.date_sub(F.lit(selected_date), 30)
   end_plot_date = F.date_add(F.lit(selected_date), 30)
   
   chart_data = test_spark_df.filter(
                                 (F.col("Date") >= start_plot_date) & 
                                 (F.col("Date") <= end_plot_date)
                           ) \
                           .dropDuplicates(["Date"]) \
                           .orderBy("Date")
   # Lấy dữ liệu gốc (Chứa Date và các cột Target thực tế)
   pdf_plot = chart_data.toPandas()

   if not pdf_plot.empty:
      # --- LOGIC DỰ BÁO VÀ MERGE ---
      
      # 1. Xử lý MinTemp
      key_min = f"{algo_prefix}_Target_MinTemp_D1"
      if key_min in models:
         # Predict trên Spark
         res = models[key_min].transform(chart_data) \
                              .select("Date", "prediction") \
                              .withColumnRenamed("prediction", "Pred_Min")
         # Convert sang Pandas
         pdf_res = res.toPandas()
         # Merge vào bảng chính (Chỉ merge cột Pred_Min)
         pdf_plot = pd.merge(pdf_plot, pdf_res, on="Date", how="left")
      else:
         # Nếu không có model, điền số 0 để không lỗi code vẽ
         pdf_plot["Pred_Min"] = 0.0

      # 2. Xử lý MaxTemp
      key_max = f"{algo_prefix}_Target_MaxTemp_D1"
      if key_max in models:
         res = models[key_max].transform(chart_data) \
                              .select("Date", "prediction") \
                              .withColumnRenamed("prediction", "Pred_Max")
         pdf_res = res.toPandas()
         pdf_plot = pd.merge(pdf_plot, pdf_res, on="Date", how="left")
      else:
         pdf_plot["Pred_Max"] = 0.0

      # 3. Xử lý Rainfall
      key_rain = f"{algo_prefix}_Target_Rainfall_D1"
      if key_rain in models:
         res = models[key_rain].transform(chart_data) \
                                 .select("Date", "prediction") \
                                 .withColumnRenamed("prediction", "Pred_Rain")
         pdf_res = res.toPandas()
         pdf_plot = pd.merge(pdf_plot, pdf_res, on="Date", how="left")
      else:
         pdf_plot["Pred_Rain"] = 0.0
      
      # --- VẼ BIỂU ĐỒ (MATPLOTLIB) ---
      import matplotlib.dates as mdates
      
      # Tạo 2 biểu đồ con (Subplots)
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
      
      # --- BIỂU ĐỒ 1: NHIỆT ĐỘ ---
      # Kiểm tra cột tồn tại trước khi vẽ để chắc chắn
      if 'Target_MinTemp_D1' in pdf_plot.columns:
         ax1.plot(pdf_plot['Date'], pdf_plot['Target_MinTemp_D1'], label="Min Thực tế", color='blue', linestyle='-', alpha=0.5)
      ax1.plot(pdf_plot['Date'], pdf_plot['Pred_Min'], label="Min Dự báo", color='navy', linestyle='--', linewidth=2)
      
      if 'Target_MaxTemp_D1' in pdf_plot.columns:
         ax1.plot(pdf_plot['Date'], pdf_plot['Target_MaxTemp_D1'], label="Max Thực tế", color='orange', linestyle='-', alpha=0.5)
      ax1.plot(pdf_plot['Date'], pdf_plot['Pred_Max'], label="Max Dự báo", color='red', linestyle='--', linewidth=2)
      
      ax1.set_title(f"Phân tích Bối cảnh (Trước/Sau 30 ngày) - {model_type}", fontweight='bold')
      
      # Thêm một đường kẻ dọc để đánh dấu ngày hiện tại (Ngày T)
      ax1.axvline(x=selected_date, color='black', linestyle='-', linewidth=1, label="Ngày được chọn")
      ax2.axvline(x=selected_date, color='black', linestyle='-', linewidth=1)
      ax1.set_ylabel("Nhiệt độ (°C)")
      ax1.legend(loc="upper left")
      ax1.grid(True, linestyle=':', alpha=0.5)
      
      # --- BIỂU ĐỒ 2: LƯỢNG MƯA ---
      if 'Target_Rainfall_D1' in pdf_plot.columns:
         ax2.plot(pdf_plot['Date'], pdf_plot['Target_Rainfall_D1'], label="Mưa Thực tế", color='#1f77b4', alpha=0.6)
      ax2.plot(pdf_plot['Date'], pdf_plot['Pred_Rain'], label="Mưa Dự báo", color='green', linestyle='--', linewidth=2)
      
      ax2.set_title(f"Dự báo Lượng mưa - {model_type}", fontweight='bold')
      ax2.set_ylabel("Lượng mưa (mm)")
      ax2.legend(loc="upper left")
      ax2.grid(True, linestyle=':', alpha=0.5)

      # Format ngày tháng
      ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
      plt.xticks(rotation=45)
      
      st.pyplot(fig)
      
      # Bảng dữ liệu thô (Optional)
      with st.expander("Xem bảng số liệu chi tiết"):
         cols_to_show = ['Date', 'Pred_Min', 'Pred_Max', 'Pred_Rain']
         # Chỉ lấy các cột Target nếu nó tồn tại trong file test
         for t in ['Target_MinTemp_D1', 'Target_MaxTemp_D1', 'Target_Rainfall_D1']:
               if t in pdf_plot.columns:
                  cols_to_show.insert(1, t)
         st.dataframe(pdf_plot[cols_to_show])
         
   else:
      st.warning("Không đủ dữ liệu để vẽ biểu đồ.")
if __name__ == "__main__":
   main()