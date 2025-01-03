# FBCNN-GUI: A User-Friendly Interface for Flexible Blind JPEG Artifacts Removal

[![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/YOUR_REPO_NAME?style=social)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)

This project, **FBCNN-GUI**, provides an easy-to-use graphical user interface for [**FBCNN**](https://github.com/jiaxi-jiang/FBCNN) (Towards Flexible Blind JPEG Artifacts Removal). FBCNN is a state-of-the-art deep learning model for removing artifacts from JPEG compressed images. This GUI allows users to easily load models, process images, and view results through intuitive operations, without the need for complex command-line instructions.

## ✨ Key Features

* **User-Friendly Graphical Interface:**  Complete image import, model selection, processing, and result viewing through simple window operations.
* **Flexible Model Selection:** Supports loading different models provided by FBCNN to suit various processing needs.
* **Real-time Preview:** Easily compare original and processed images to evaluate the de-artifacting effect.
* **Batch Processing:** Supports processing multiple image files at once.
* **One-Click Installation (Optional):** The provided `1.install&run.bat` script allows for quick environment configuration and program launch.

## 🚀🚀 Some Visual Examples (Click to view the full image)

| [![Original Image](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/kodak24_bpp0.746.png)](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/kodak24_bpp0.746.png) | [![FBCNN-D Result](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/kodak24_bpp0.746_fbcnn_d.png)](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/kodak24_bpp0.746_fbcnn_d.png) |
|---|---|
| [![Original Image](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/kodak03_bpp0.467.png)](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/kodak03_bpp0.467.png) | [![FBCNN-A Result](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/kodak03_bpp0.467_fbcnn_a.png)](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/kodak03_bpp0.467_fbcnn_a.png) |

---

## Introduction to FBCNN Principles (Cited from [jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN))

### Motivations

JPEG is one of the most widely used image compression algorithms and formats due to its simplicity and fast encoding/decoding speed. However, it is a lossy compression algorithm that may introduce annoying artifacts. Existing JPEG artifact removal methods often have four limitations in practical applications:

* Most existing learning-based methods \[e.g., ARCNN, MWCNN, SwinIR] train a specific model for each quality factor, lacking the flexibility of learning a single model for different JPEG quality factors.
* DCT-based methods \[e.g., DMCNN, QGAC] require obtaining DCT coefficients or quantization tables as input, which are only stored in the JPEG format. Moreover, when an image is compressed multiple times, only the most recent compression information is stored.
* Existing blind methods \[e.g., DnCNN, DCSC, QGAC] can only provide deterministic reconstruction results for each input, ignoring the needs of user preference.
* Existing methods are trained with synthetic images, assuming that low-quality images are only compressed once. **However, most images from the Internet are compressed multiple times.** Although some progress has been made on real recompressed images, such as images from Twitter \[ARCNN, DCSC], there is still a lack of detailed and complete research on double JPEG artifact removal.

### Network Architecture

We propose a flexible blind convolutional neural network (FBCNN) that can predict the quality factor of a JPEG image and embed it into the decoder to guide image restoration. The quality factor can be manually adjusted for flexible JPEG restoration according to user preference. [architecture](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/architecture.png)

### Analysis of Double JPEG Restoration

#### 1. What is Unaligned Double JPEG Compression?

Unaligned double JPEG compression means that the 8x8 blocks of the two JPEG compressions are not aligned. For example, when we crop a JPEG image and save it as JPEG, it is very likely to get an unaligned double JPEG image. [real](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/real.png) There are many other common scenarios, including but not limited to:

* Take a photo with a smartphone and upload it to the Internet. Most social media platforms, such as WeChat, Twitter, and Facebook, will resize the uploaded image by downsampling and then apply JPEG compression to save storage space.
* Edit a JPEG image that introduces cropping, rotation, or resizing, and save it as JPEG.
* Zoom in/out of a JPEG image, then take a screenshot and save it as JPEG.
* Group different JPEG images and save them as a single JPEG image.
* Most memes are compressed multiple times and the situation is unaligned.

#### 2. Limitations of Existing Blind Methods in Restoring Unaligned Double JPEG Images

We found that when the 8x8 blocks of the two JPEG compressions are unaligned and QF1 <= QF2, existing blind methods always fail even if there is only **one pixel shift**. Other situations, such as unaligned double JPEG with QF1>QF2 or aligned double JPEG compression, are actually equivalent to single JPEG compression.

Here is an example of the restoration results of JPEG images by DnCNN and QGAC under different degradation settings. '*' means there is a one-pixel shift between the two JPEG blocks.

![lena_doublejpeg](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/figs/lena_doublejpeg.png)

#### 3. Our Solutions

We found that in unaligned double JPEG images with QF1 < QF2, FBCNN always predicts the quality factor as QF2. However, it is the smaller QF1 that dominates the compression artifacts. By manually changing the predicted quality factor to QF1, we can largely improve the results.

Furthermore, to obtain a completely blind model, we propose two blind solutions to address this issue:

(1) FBCNN-D: Train the model using a single JPEG degradation model + automatic dominant QF correction. By leveraging the characteristics of JPEG images, we found that the quality factor of a single JPEG image can be predicted by applying another JPEG compression. The MSE of the two JPEG images is the smallest when QF1 = QF2. In our paper, we also extend this method to the unaligned double JPEG case to obtain a completely blind model.

(2) FBCNN-A: Augment the training data using a double JPEG degradation model, which is given by the formula:

y = JPEG(shift(JPEG(x, QF1)),QF2)


By mitigating the misalignment between the training data and real-world JPEG images, FBCNN-A further improves the performance of complex double JPEG restoration. **The proposed double JPEG degradation model can be easily integrated into other image restoration tasks, such as single image super-resolution (e.g., BSRGAN), to achieve better general real-world image restoration performance.**

## 🛠️ Installation

### One-Click Installation

1. Ensure you have Git installed.
2. Download or clone this repository to your local machine.
3. Double-click the `1.install&run.bat` file in the repository's root directory.

   This script will automatically perform the following actions:
    * Check and install Python if it is not installed.
    * Create and activate a virtual environment named `fbcnn_env`.
    * Check and install PyTorch if it is not installed, prompting you to choose between the CPU or GPU version.
    * Set up GitHub and Hugging Face proxies to accelerate downloads.
    * Configure pip to use the Tsinghua mirror source.
    * Install the necessary dependency packages for the project.
    * Run the GUI program (`gui.py`).

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
6. **Run the GUI:**
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

## Citation

@inproceedings{jiang2021towards,
title={Towards Flexible Blind JPEG Artifacts Removal},
author={Jiang, Jiaxi and Zhang, Kai and Timofte, Radu},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={4997--5006},
year={2021}
}


## License and Acknowledgement

This project is released under the Apache 2.0 license. This work was supported in part by the ETH Zurich Fund (OK) and a project from Huawei Technologies Co., Ltd (Finland).
