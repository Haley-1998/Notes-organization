为了方便程序可以在不同电脑上直接运行，可建立/转移虚拟环境至移动硬盘。

以下均在移动硬盘目录操作。电脑上普通安装和配置同样操作步骤。

## 1、安装Miniconda

[Miniconda — Anaconda documentation](https://docs.anaconda.com/free/miniconda/)

## 2、配置

 使用cmd激活conda环境
导航到移动硬盘上的 Miniconda 安装目录。

```
cd C:\Users\YourUsername\miniconda3\Scripts
activate.bat
```

ps：在当前目录选中路径输入cmd，直接进入当前路径的cmd

![image](https://github.com/user-attachments/assets/5427862a-3e50-44bc-b72c-b2ec775622c1)

### 创建虚拟环境
```
conda create -n myenv python=3.9
```
### 激活虚拟环境
```
conda activate myenv
```
### 安装Spyder
```
conda install spyder
```
### 安装库和依赖项
```
pip install numpy

or 

conda install numpy

...
```
![image](https://github.com/user-attachments/assets/b5214f83-00cc-4079-927a-7dc4c2ae9910)

## 3、为虚拟环境创建桌面快捷方式

为特定的虚拟环境创建一个直接启动 Spyder 的桌面快捷方式，可以方便地在特定虚拟环境中启动 Spyder，并确保 Spyder 使用该环境中的包和库。
```
@echo off
call C:\Users\YourUsername\miniconda3\Scripts\activate.bat myenv
spyder
```
将 C:\Users\YourUsername\miniconda3 替换为您的 Anaconda 安装路径（到Scripts），将 myenv 替换为您的虚拟环境名称。

将文件保存为 .bat 文件，例如 launch_spyder_myenv.bat。

双击.bat 文件，激活指定的虚拟环境并启动 Spyder。

## PS：

1、不同电脑上的移动硬盘目录可能不一样。

2、环境变量配置

## 4、库转移（重新批量下载）
 进入旧环境
```
activate.bat

activate tensorflow
```
使用pip freeze命令导出当前环境中的包列表，并保存到一个文本文件中。
```
pip list --format=freeze > requirements.txt

dir /s /b requirements.txt        #查看文件位置
```
### 进入新环境
```
activate newenv
```
### 安装包到新环境中：
```
pip install -r requirements.txt --no-dependencies
```
PS：有版本冲突的修改requirements.txt中的版本或者pip install单独重新安装(指定版本/不指定)

## 其他

查看虚拟环境目录
```
where python
```
列出所有包版本信息
```
pip list
```

