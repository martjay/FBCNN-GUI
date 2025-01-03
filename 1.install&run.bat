@echo off

chcp 65001

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed, downloading and installing...
    powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('https://mirrors.huaweicloud.com/python/3.11.5/python-3.11.5-amd64.exe', 'python-installer.exe') }"
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-installer.exe
)

REM Check if the virtual environment exists, if not, create it
if not exist "fbcnn_env" (
    python -m venv fbcnn_env
)

REM Activate the virtual environment
call fbcnn_env\Scripts\activate.bat

REM Check if PyTorch is installed
python -c "import torch; print(torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    REM PyTorch is not installed, prompt the user to select the installation version
    echo Please select the installation version:
    echo 1. CPU version
    echo 2. GPU version
    set /p version_choice="Enter the number and press Enter (1 or 2): "

    echo Your input choice is: [%version_choice%]

    if "%version_choice%"=="1" (
        echo Installing the CPU version of PyTorch...
        pip install torch torchvision torchaudio
    ) else if "%version_choice%"=="2" (
        echo Installing the GPU version of PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo Invalid selection, installing the GPU version by default...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo PyTorch is installed, skipping the installation step.
)

REM Set GitHub proxy
set "GIT_PROXY_COMMAND=kkgithub.com"

REM Set HuggingFace proxy
set "HF_PROXY_COMMAND=hf-mirror.com"

REM Set pip to use Tsinghua mirror
echo Setting pip to use Tsinghua mirror...
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

REM Install dependencies
pip install -r requirements.txt

REM Run your Python script or command
python gui.py

REM Deactivate the virtual environment
deactivate
