# FBCNN-GUI: A User-Friendly Interface for Flexible Blind JPEG Artifacts Removal

[![GitHub Stars](https://img.shields.io/github/stars/martjay/FBCNN-GUI?style=social)](https://github.com/martjay/FBCNN-GUI)

This project, **FBCNN-GUI**, provides an easy-to-use graphical user interface for [**FBCNN**](https://github.com/jiaxi-jiang/FBCNN) (Towards Flexible Blind JPEG Artifacts Removal). FBCNN is a state-of-the-art deep learning model for removing artifacts from JPEG compressed images. This GUI allows users to easily load models, process images, and view results through intuitive operations, without the need for complex command-line instructions.

## ✨ Key Features

* **User-Friendly Graphical Interface:**  Complete image import, model selection, processing, and result viewing through simple window operations.
* **Flexible Model Selection:** Supports loading different models provided by FBCNN to suit various processing needs.
* **Real-time Preview:** Easily compare original and processed images to evaluate the de-artifacting effect.
* **Batch Processing:** Supports processing multiple image files at once.
* **One-Click Installation (Optional):** The provided `1.install&run.bat` script allows for quick environment configuration and program launch.

🚀🚀 Some Visual Examples (Click for full images)
----------
![Image description](https://raw.githubusercontent.com/martjay/FBCNN-GUI/refs/heads/main/picture1.png)

![Image description](https://raw.githubusercontent.com/martjay/FBCNN-GUI/refs/heads/main/picture2.png)

---

## Introduction to FBCNN Principles (Cited from [jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN))

### Manual Installation

1. **Install Python:** Ensure you have Python installed on your system (Python 3.7 or later is recommended). You can download and install it from [python.org](https://www.python.org/).
2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv fbcnn_env
    ```
3. **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        fbcnn_env\Scripts\activate
        ```
    *   **Linux/macOS:**
        ```bash
        source fbcnn_env/bin/activate
        ```
4. **Install PyTorch:** Install PyTorch based on your hardware. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation commands. For example, to install the CPU version:
    ```bash
    pip install torch torchvision torchaudio
    ```
    Or to install the CUDA 11.8 GPU version:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
5. **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6. **Download the following models from [Github Release](https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0).**
    * fbcnn_color.pth 
    * fbcnn_gray.pth
    * fbcnn_gray_double.pth

7. **Run the GUI:**
    ```bash
    python gui.py
    ```

## ⚙️ Usage

1. **Start the Program:** Run the `gui.py` script.
2. **Select a Model:** Choose the desired FBCNN model from the dropdown menu at the top of the interface.
3. **Select a Device:** Choose whether to use CPU or GPU for processing.
4. **Import Images:** Click the "Import Images" button to select the image files you want to process.
5. **Start Processing:** Click the "Start Processing" button to begin processing the selected images.
6. **View Results:** The processed image will be displayed on the interface. You can switch between viewing the original and processed images.
7. **Save Results:** You can save the currently processed image individually or save all processed images in batch.


## License and Acknowledgement

This project is released under the Apache 2.0 license. This work was supported in part by the ETH Zurich Fund (OK) and a project from Huawei Technologies Co., Ltd (Finland).
