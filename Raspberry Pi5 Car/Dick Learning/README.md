若要在本地訓練、推理YOLOv11 seg模型，及在Raspberry Pi上使用模型，請照下列步驟安裝環境。

在本地部屬YOLOv11流程

去到:
https://github.com/ultralytics/ultralytics#

<>Code -> Download zip 後解壓縮到你要存放的路徑

下載Anaconda (miniconda) or uv等python虛擬環境管理工具，這邊以miniconda為例示範。<br>
安裝完成miniconda後啟動Anaconda Prompt創立一個YOLO用的虛擬環境<br>
```md
conda create -n YOLO python = 3.11 
```
創建完成後我們要下載Pytorch，Pytorch根據你的Nvidia GPU型號來下載對應支持的Pytorch版本<br>
若使用的是CPU的話就下載CPU版本的Pytorch，這邊不對Pytorch Cuda版本的查詢進行教學<br>
可以到Nvidia驅動面板中察看或自行上網搜尋，不同版本Pytorch安裝指令請到: https://pytorch.org/get-started/locally/ 進行查詢<br>

這邊以GPU型號 : RTX 5080 作為範例(50系列顯卡可以照抄)
安裝Pytorch CUDA 12.8版本<br>
```md
conda activate YOLO
``` 

```md
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安裝完成後請確認Pytorch Cuda可以正常使用再繼續操作<br>
找到你存放ultralytics的資料夾，複製該資料夾地址<br>
在Anaconda Prompt中輸入

```md
pushd #你的地址
```

```md
pip install -e .
```
去完成YOLO的安裝。



