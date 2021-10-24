写在前面，利用TensorRT加速推理速度是以时间换取精度的做法，意味着在推理速度上升的同时将会有精度的下降，不过不用太担心，精度下降微乎其微。此外，要有NVIDIA显卡，经测试，CUDA10.2可以支持20系列显卡及以下，30系列显卡需要CUDA11.x的支持，并且目前有bug。

默认你已经完成了 yolov5的训练过程并得到了.pt模型权值文件。

本文目的仅是带着走通流程。

注意要对应yolov5和tensorrtx的版本。

### 我的运行环境(注意OpenCV要选择适合你的visual studio的版本等问题)：
```
win10

Visual Studio 2019

NVIDIA GeForce RTX 2060

opencv-3.4.3-vc14_vc15

cuda_10.2.89_441.22_win10

cudnn-10.2-windows10-x64-v7.6.5.32

TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6

cmake-3.21.2-windows-x86_64
```

上述环境的百度云(测试10、20系列可用)：
```
链接：https://pan.baidu.com/s/1AyaloTzLap8X2hsJBvyeBw
提取码：dwr7
```
其他版本下载地址：

CUDA  cudnn  TensorRT  CMake  OpenCV
 

## 环境安装：

1、安装OpenCV并配置好环境变量

2、安装CUDA

    一路默认。一般的安装路径为：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2

3、安装cudnn和TensorRT

    cudnn和TensorRT的安装仅是将下载的对应版本的压缩包解压并复制*.h、*.lib、*.dll到CUDA的安装路径。

    1 将cuDNN压缩包解压

    2 将cuda\bin中的文件复制到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin

    3 将cuda\include中的文件复制到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include

    4 将cuda\lib中的文件复制到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib

    另外，

    1 将TensorRT压缩包解压

    2 将 TensorRT-7.0.0.11\include中头文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include

    3 将TensorRT-7.0.0.11\lib中所有lib文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64

    4 将TensorRT-7.0.0.11\lib中所有dll文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin

4、安装CMake软件备用

## 一、将训练阶段得到的.pt模型转化为.wts中间模型

把tensorrtx里面的yolov5\gen_wts.py加入到yolov5里面，执行
```
python gen_wts.py -w [.pt权值文件路径] 
```
runs\train\exp\weights\best.pt为训练过程生成的.pt模型，生成的best.wts会保存到同目录下，此best.wts待会会用到。

cuda版本每个电脑不一样

配置好的tensorrtx，包括Cmakelist.txt的设定以及dirent.h的配置。

若使用原作者的请参照tensorrtx源码https://github.com/wang-xinyu/tensorrtx ，配置过程中会遇到一些问题，挨个解决，问题不大。

1、在yolov5目录下新建build文件夹

2、修改CMakelist.txt
```
add_definitions(-DAPI_EXPORTS)
```
3、打开CMake

​​
generate后关闭

4、yolov5/include/dirent.h

​​

也可使用配置好的 我的

## 二、利用Cmake软件创建VS工程

修改CMakeLists.txt中此处为你的opencv安装路径。

配置好上方两个目录之后，点击Configure，根据你的环境选择配置，

点击Gnerate，警告可忽视，

现在关闭Cmake即可。

## 三、wts转化为engine

VS打开刚刚在bulid目录下创建的工程。

build处vs打开，生成

问题:我的模型只识别一个类，需要更改

cd {tensorrtx}/yolov5/

// update CLASS_NUM in yololayer.h if your model is trained on custom dataset

为1

生成项目。

把之前生成的best.wts复制到build\release目录里面

cmd里面运行：
```
.\test.exe -s .\best.wts best.engine s
```
运行成功在同文件夹下面会得到best.engine转换后的文件。之后的推理过程使用的都是这个文件。

测试：
```
.\yolov5.exe -d best.engine .\samples
```
至此，流程走完。

如果想要进一步封装，可以按照我的示例。

注释掉yolov5.cpp，并取消 几个文件的注释。重新生成项目。按照你的需求更改。
