## 📘 中文：YOLO 目标检测项目

本项目使用 YOLOv8 模型实现对视频和图像数据中的目标检测任务。

### 📁 项目结构

项目包含以下文件和目录：

- `.gitignore`  
- `README.md`  
- `requirements.txt`：项目依赖项列表  
- `YOLO_inference.py`：使用训练好的 YOLOv8 模型对视频文件进行推理的代码  
- `YOLO_lossVisualization.py`  
- `YOLO_train.py`：用于训练和评估 YOLOv8 模型的代码  
- `data/`：包含数据相关的文件  
  - `data_preprocessing.py`：视频和图像数据的预处理代码  
- `model/`：模型文件  
  - `4class.pt`：可识别关闭盒子、打开盒子、食物和手  
  - `11class.pt`：可识别普通、红、黑、白盒子并区分开合状态，食物为普通/辣的炸鸡和辣炒年糕  
  - `9class.pt`：与11类模型相比仅能区分食物  
  - `yolov8nOG.pt`：原始 YOLOv8n 模型  

### 🧩 依赖项

项目依赖以下 Python 库：

- ultralytics  
- cv2  
- lap  
- pandas  
- matplotlib  

安装方式：

```bash
pip install -r requirements.txt
```

### 🛠️ 使用方法

#### 1. 数据准备

1. 将原始视频数据放入 `data/raw_data/` 文件夹中。  
2. 运行以下命令预处理为图像帧：  

   ```bash
   python data/data_preprocessing.py
   ```

   - 提取帧图像  
   - 可选图像增强  
   - 保存到 `data/image_data/<video_name>/`  

3. 创建 `data/labelled_data4OBJ/` 目录，YOLO 格式标注图像：  

```
labelled_data4OBJ/
  ├── images/
  ├── labels/
  ├── classes.txt
  └── notes.json
```

#### 2. 模型训练

1. 编辑 `YOLO_train.py` 设置参数（epoch、batch size、learning rate 等）  
2. 启动训练：

```bash
python YOLO_train.py
```

此脚本将：

- 数据集划分  
- 生成 `data.yaml` 配置  
- 模型训练和评估  
- 结果保存至 `model/`  

#### 3. 模型推理

1. 设置 `YOLO_inference.py` 中要推理的视频路径  
2. 执行：

```bash
python YOLO_inference.py
```

脚本将：

- 加载训练模型  
- 运行检测并可视化结果  