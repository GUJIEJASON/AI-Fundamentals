运行：

python yolov8_CS2_aimbot.py --view-img

python yolov8_ssjj_aimbot.py

PS:生死狙击不能小窗，所以不用开view_img功能。

yolov8_ssjj_aimbot.py中的159行的bounding_box可能需要根据自己的显示器情况进行修改（CS2有自动识别，不用改）

生死狙击鼠标移动调的函数与CS2不同，需要将鼠标输入模式（进游戏后Esc设置->图像->鼠标输入模式）由原始输入模式改成Flash兼容模式。

测试：

yolo predict model="./weight/v8s_180.pt" source="./sample"

weight文件夹里有四种模型，ppt中演示的是v8s_100.pt和v8s_180.pt

sample文件夹中是测试数据集（网上找的以及自己截的）

训练数据集：[demModels - v3 2023-10-16 8:35am (roboflow.com)](https://universe.roboflow.com/sprite-fanta-gpj4f/demmodels/dataset/3)