@echo off

chcp 65001

REM Activate the virtual environment
call fbcnn_env\Scripts\activate.bat

REM Check if PyTorch is installed
python -c "import torch; print(torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo PyTorch is not installed.
) else (
    echo PyTorch is installed, version information is as follows:
    python -c "import torch; print('Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
    
    REM Prompt the user whether to uninstall the current version
    set /p uninstall_choice="Uninstall the current version of PyTorch? (Press Enter to confirm uninstallation): "
    if "%uninstall_choice%"=="" (
        echo Uninstalling the current version of PyTorch...
        pip uninstall -y torch torchvision torchaudio
    ) else (
        echo Keeping the current version of PyTorch.
        exit /b
    )
)

REM Prompt the user to select the installation version
echo Please select the installation version:
echo 1. CPU version
echo 2. GPU version
set /p version_choice="Enter the number and press Enter (1 or 2): "

if "%version_choice%"=="1" (
    echo Installing the CPU version of PyTorch...
    pip install torch torchvision torchaudio
) else if "%version_choice%"=="2" (
    echo Installing the GPU version of PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Invalid selection, installing the CPU version by default...
    pip install torch torchvision torchaudio
)

REM Deactivate the virtual environment
deactivate

REM Remind the user and wait for Enter to exit
echo Installation complete. Press Enter to exit.
pause >nul