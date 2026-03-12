# Cropper: Vision-Language Model for Image Cropping through In-Context Learning

A training-free image cropping framework using Vision-Language Models (VLMs) with in-context learning. This is a replication of the CVPR 2025 paper using open-source VLMs (Mantis-8B-Idefics2) instead of Gemini 1.5 Pro.

## Overview
(cropper_env) es22btech11013@aicoe-a6000:~/divya/AFCIL/divya/cv-project/cropper$ pip install -r requirements.txt
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: torch>=2.0.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 2)) (2.9.0)
Requirement already satisfied: torchvision>=0.15.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 3)) (0.24.0)
Collecting transformers>=4.36.0
  Downloading transformers-5.3.0-py3-none-any.whl (10.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.7/10.7 MB 29.9 MB/s eta 0:00:00
Collecting accelerate>=0.25.0
  Downloading accelerate-1.13.0-py3-none-any.whl (383 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 383.7/383.7 KB 51.7 MB/s eta 0:00:00
Collecting mantis-vl>=0.0.1
  Downloading mantis_vl-0.0.5-py3-none-any.whl (376 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 376.0/376.0 KB 42.6 MB/s eta 0:00:00
Collecting open-clip-torch>=2.20.0
  Downloading open_clip_torch-3.3.0-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 69.2 MB/s eta 0:00:00
Collecting Pillow>=10.0.0
  Downloading pillow-12.1.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 29.9 MB/s eta 0:00:00
Requirement already satisfied: opencv-python>=4.8.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 15)) (4.12.0.88)
Requirement already satisfied: numpy>=1.24.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 18)) (2.2.6)
Collecting pandas>=2.0.0
  Downloading pandas-2.3.3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.8/12.8 MB 49.6 MB/s eta 0:00:00
Collecting scipy>=1.11.0
  Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 24.8 MB/s eta 0:00:00
Collecting PyYAML>=6.0
  Downloading pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (770 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 770.3/770.3 KB 17.5 MB/s eta 0:00:00
Collecting omegaconf>=2.3.0
  Downloading omegaconf-2.3.0-py3-none-any.whl (79 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.5/79.5 KB 27.0 MB/s eta 0:00:00
Requirement already satisfied: tqdm>=4.65.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 27)) (4.67.1)
Collecting rich>=13.0.0
  Downloading rich-14.3.3-py3-none-any.whl (310 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 310.5/310.5 KB 73.4 MB/s eta 0:00:00
Collecting scikit-learn>=1.3.0
  Downloading scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.7/9.7 MB 47.6 MB/s eta 0:00:00
Collecting faiss-cpu>=1.7.4
  Downloading faiss_cpu-1.13.2-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (23.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.8/23.8 MB 35.8 MB/s eta 0:00:00
Collecting h5py>=3.9.0
  Downloading h5py-3.16.0-cp310-cp310-manylinux_2_28_x86_64.whl (5.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 49.2 MB/s eta 0:00:00
Requirement already satisfied: matplotlib>=3.7.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from -r requirements.txt (line 38)) (3.10.7)
Collecting seaborn>=0.12.0
  Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 KB 67.8 MB/s eta 0:00:00
Collecting pytest>=7.0.0
  Downloading pytest-9.0.2-py3-none-any.whl (374 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 374.8/374.8 KB 79.7 MB/s eta 0:00:00
Collecting black>=23.0.0
  Downloading black-26.3.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 77.0 MB/s eta 0:00:00
Collecting isort>=5.12.0
  Downloading isort-8.0.1-py3-none-any.whl (89 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.7/89.7 KB 29.1 MB/s eta 0:00:00
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.4.1)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (3.3.20)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (2.27.5)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.93)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (11.3.3.83)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (9.10.2.21)
Requirement already satisfied: triton==3.5.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (3.5.0)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.90)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.90)
Collecting fsspec>=0.8.5
  Downloading fsspec-2026.2.0-py3-none-any.whl (202 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 202.5/202.5 KB 37.5 MB/s eta 0:00:00
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.90)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.8.93)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (12.5.8.93)
Requirement already satisfied: jinja2 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (3.1.6)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (10.3.9.90)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (0.7.1)
Requirement already satisfied: sympy>=1.13.3 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (1.14.0)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (11.7.3.90)
Requirement already satisfied: filelock in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (3.20.0)
Requirement already satisfied: networkx>=2.5.1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (3.4.2)
Requirement already satisfied: typing-extensions>=4.10.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (4.15.0)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from torch>=2.0.0->-r requirements.txt (line 2)) (1.13.1.3)
Collecting typer
  Downloading typer-0.24.1-py3-none-any.whl (56 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.1/56.1 KB 13.7 MB/s eta 0:00:00
Collecting safetensors>=0.4.3
  Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (507 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 507.2/507.2 KB 53.0 MB/s eta 0:00:00
Collecting huggingface-hub<2.0,>=1.3.0
  Downloading huggingface_hub-1.6.0-py3-none-any.whl (612 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 612.9/612.9 KB 90.9 MB/s eta 0:00:00
Collecting tokenizers<=0.23.0,>=0.22.0
  Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 58.8 MB/s eta 0:00:00
Collecting regex!=2019.12.17
  Downloading regex-2026.2.28-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (791 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 791.7/791.7 KB 25.9 MB/s eta 0:00:00
Requirement already satisfied: packaging>=20.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from transformers>=4.36.0->-r requirements.txt (line 4)) (25.0)
Requirement already satisfied: psutil in /usr/lib/python3/dist-packages (from accelerate>=0.25.0->-r requirements.txt (line 5)) (5.9.0)
Collecting datasets==2.18.0
  Downloading datasets-2.18.0-py3-none-any.whl (510 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 510.5/510.5 KB 73.1 MB/s eta 0:00:00
Collecting sentencepiece
  Downloading sentencepiece-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (1.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 73.8 MB/s eta 0:00:00
Collecting pyarrow-hotfix
  Downloading pyarrow_hotfix-0.7-py3-none-any.whl (7.9 kB)
Collecting xxhash
  Downloading xxhash-3.6.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (193 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 193.2/193.2 KB 54.5 MB/s eta 0:00:00
Collecting dill<0.3.9,>=0.3.0
  Downloading dill-0.3.8-py3-none-any.whl (116 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 116.3/116.3 KB 43.9 MB/s eta 0:00:00
Collecting fsspec[http]<=2024.2.0,>=2023.1.0
  Downloading fsspec-2024.2.0-py3-none-any.whl (170 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.9/170.9 KB 46.6 MB/s eta 0:00:00
Collecting multiprocess
  Downloading multiprocess-0.70.19-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.9/134.9 KB 43.7 MB/s eta 0:00:00
Collecting pyarrow>=12.0.0
  Downloading pyarrow-23.0.1-cp310-cp310-manylinux_2_28_x86_64.whl (47.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.6/47.6 MB 23.7 MB/s eta 0:00:00
Collecting aiohttp
  Downloading aiohttp-3.13.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 33.9 MB/s eta 0:00:00
Requirement already satisfied: requests>=2.19.0 in /usr/lib/python3/dist-packages (from datasets==2.18.0->mantis-vl>=0.0.1->-r requirements.txt (line 8)) (2.25.1)
Collecting ftfy
  Downloading ftfy-6.3.1-py3-none-any.whl (44 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 KB 15.4 MB/s eta 0:00:00
Collecting timm>=1.0.17
  Downloading timm-1.0.25-py3-none-any.whl (2.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 MB 46.8 MB/s eta 0:00:00
Requirement already satisfied: python-dateutil>=2.8.2 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from pandas>=2.0.0->-r requirements.txt (line 19)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas>=2.0.0->-r requirements.txt (line 19)) (2022.1)
Collecting tzdata>=2022.7
  Downloading tzdata-2025.3-py2.py3-none-any.whl (348 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 348.5/348.5 KB 74.3 MB/s eta 0:00:00
Collecting antlr4-python3-runtime==4.9.*
  Downloading antlr4-python3-runtime-4.9.3.tar.gz (117 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.0/117.0 KB 25.9 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting markdown-it-py>=2.2.0
  Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.3/87.3 KB 15.1 MB/s eta 0:00:00
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from rich>=13.0.0->-r requirements.txt (line 28)) (2.19.2)
Collecting threadpoolctl>=3.1.0
  Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Collecting joblib>=1.2.0
  Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 309.1/309.1 KB 72.5 MB/s eta 0:00:00
Requirement already satisfied: kiwisolver>=1.3.1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from matplotlib>=3.7.0->-r requirements.txt (line 38)) (1.4.9)
Requirement already satisfied: fonttools>=4.22.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from matplotlib>=3.7.0->-r requirements.txt (line 38)) (4.60.1)
Requirement already satisfied: cycler>=0.10 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from matplotlib>=3.7.0->-r requirements.txt (line 38)) (0.12.1)
Requirement already satisfied: pyparsing>=3 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from matplotlib>=3.7.0->-r requirements.txt (line 38)) (3.2.5)
Requirement already satisfied: contourpy>=1.0.1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from matplotlib>=3.7.0->-r requirements.txt (line 38)) (1.3.2)
Requirement already satisfied: exceptiongroup>=1 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from pytest>=7.0.0->-r requirements.txt (line 50)) (1.3.0)
Collecting pluggy<2,>=1.5
  Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
Collecting iniconfig>=1.0.1
  Downloading iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Collecting tomli>=1
  Downloading tomli-2.4.0-py3-none-any.whl (14 kB)
Requirement already satisfied: platformdirs>=2 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from black>=23.0.0->-r requirements.txt (line 51)) (4.5.0)
Collecting mypy-extensions>=0.4.3
  Downloading mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
Collecting pytokens~=0.4.0
  Downloading pytokens-0.4.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (259 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 259.5/259.5 KB 62.5 MB/s eta 0:00:00
Requirement already satisfied: click>=8.0.0 in /usr/lib/python3/dist-packages (from black>=23.0.0->-r requirements.txt (line 51)) (8.0.3)
Collecting pathspec>=1.0.0
  Downloading pathspec-1.0.4-py3-none-any.whl (55 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.2/55.2 KB 19.2 MB/s eta 0:00:00
Collecting httpx<1,>=0.23.0
  Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.5/73.5 KB 11.9 MB/s eta 0:00:00
Collecting hf-xet<2.0.0,>=1.3.2
  Downloading hf_xet-1.3.2-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/4.2 MB 60.6 MB/s eta 0:00:00
Collecting mdurl~=0.1
  Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->-r requirements.txt (line 19)) (1.16.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 2)) (1.3.0)
Requirement already satisfied: wcwidth in /data1/es22btech11013/.local/lib/python3.10/site-packages (from ftfy->open-clip-torch>=2.20.0->-r requirements.txt (line 11)) (0.2.14)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/lib/python3/dist-packages (from jinja2->torch>=2.0.0->-r requirements.txt (line 2)) (2.0.1)
Collecting shellingham>=1.3.0
  Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Collecting annotated-doc>=0.0.2
  Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Collecting click>=8.0.0
  Downloading click-8.3.1-py3-none-any.whl (108 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 108.3/108.3 KB 34.6 MB/s eta 0:00:00
Collecting propcache>=0.2.0
  Downloading propcache-0.4.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (196 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.9/196.9 KB 47.8 MB/s eta 0:00:00
Collecting aiohappyeyeballs>=2.5.0
  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Requirement already satisfied: attrs>=17.3.0 in /data1/es22btech11013/.local/lib/python3.10/site-packages (from aiohttp->datasets==2.18.0->mantis-vl>=0.0.1->-r requirements.txt (line 8)) (25.4.0)
Collecting yarl<2.0,>=1.17.0
  Downloading yarl-1.23.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (102 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 102.8/102.8 KB 28.0 MB/s eta 0:00:00
Collecting multidict<7.0,>=4.5
  Downloading multidict-6.7.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (243 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.3/243.3 KB 34.8 MB/s eta 0:00:00
Collecting frozenlist>=1.1.1
  Downloading frozenlist-1.8.0-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (219 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 219.5/219.5 KB 59.6 MB/s eta 0:00:00
Collecting async-timeout<6.0,>=4.0
  Downloading async_timeout-5.0.1-py3-none-any.whl (6.2 kB)
Collecting aiosignal>=1.4.0
  Downloading aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Collecting anyio
  Downloading anyio-4.12.1-py3-none-any.whl (113 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 113.6/113.6 KB 37.8 MB/s eta 0:00:00
Collecting httpcore==1.*
  Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.8/78.8 KB 25.6 MB/s eta 0:00:00
Requirement already satisfied: idna in /usr/lib/python3/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers>=4.36.0->-r requirements.txt (line 4)) (3.3)
Requirement already satisfied: certifi in /usr/lib/python3/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.3.0->transformers>=4.36.0->-r requirements.txt (line 4)) (2020.6.20)
Collecting h11>=0.16
  Downloading h11-0.16.0-py3-none-any.whl (37 kB)
Collecting multiprocess
  Downloading multiprocess-0.70.18-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.9/134.9 KB 34.9 MB/s eta 0:00:00
  Downloading multiprocess-0.70.17-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.8/134.8 KB 42.3 MB/s eta 0:00:00
  Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.8/134.8 KB 24.7 MB/s eta 0:00:00
Building wheels for collected packages: antlr4-python3-runtime
  Building wheel for antlr4-python3-runtime (setup.py) ... done
  Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.9.3-py3-none-any.whl size=144575 sha256=897da4506e46a708d75d6cf0f0dddb0a2cf95528eb4ead1b6067c5bf03f20143
  Stored in directory: /data1/es22btech11013/.cache/pip/wheels/12/93/dd/1f6a127edc45659556564c5730f6d4e300888f4bca2d4c5a88
Successfully built antlr4-python3-runtime
Installing collected packages: antlr4-python3-runtime, xxhash, tzdata, tomli, threadpoolctl, shellingham, sentencepiece, scipy, safetensors, regex, PyYAML, pytokens, pyarrow-hotfix, pyarrow, propcache, pluggy, Pillow, pathspec, mypy-extensions, multidict, mdurl, joblib, isort, iniconfig, hf-xet, h5py, h11, ftfy, fsspec, frozenlist, faiss-cpu, dill, click, async-timeout, annotated-doc, aiohappyeyeballs, yarl, scikit-learn, pytest, pandas, omegaconf, multiprocess, markdown-it-py, httpcore, black, anyio, aiosignal, seaborn, rich, httpx, aiohttp, typer, huggingface-hub, tokenizers, timm, datasets, accelerate, transformers, open-clip-torch, mantis-vl
  WARNING: The scripts isort and isort-identify-imports are installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script ftfy is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts py.test and pytest are installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown-it is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts black and blackd are installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script httpx is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script typer is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts hf and tiny-agents are installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script datasets-cli is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts accelerate, accelerate-config, accelerate-estimate-memory, accelerate-launch and accelerate-merge-weights are installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script transformers is installed in '/data1/es22btech11013/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed Pillow-12.1.1 PyYAML-6.0.3 accelerate-1.13.0 aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-1.4.0 annotated-doc-0.0.4 antlr4-python3-runtime-4.9.3 anyio-4.12.1 async-timeout-5.0.1 black-26.3.0 click-8.3.1 datasets-2.18.0 dill-0.3.8 faiss-cpu-1.13.2 frozenlist-1.8.0 fsspec-2024.2.0 ftfy-6.3.1 h11-0.16.0 h5py-3.16.0 hf-xet-1.3.2 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-1.6.0 iniconfig-2.3.0 isort-8.0.1 joblib-1.5.3 mantis-vl-0.0.5 markdown-it-py-4.0.0 mdurl-0.1.2 multidict-6.7.1 multiprocess-0.70.16 mypy-extensions-1.1.0 omegaconf-2.3.0 open-clip-torch-3.3.0 pandas-2.3.3 pathspec-1.0.4 pluggy-1.6.0 propcache-0.4.1 pyarrow-23.0.1 pyarrow-hotfix-0.7 pytest-9.0.2 pytokens-0.4.1 regex-2026.2.28 rich-14.3.3 safetensors-0.7.0 scikit-learn-1.7.2 scipy-1.15.3 seaborn-0.13.2 sentencepiece-0.2.1 shellingham-1.5.4 threadpoolctl-3.6.0 timm-1.0.25 tokenizers-0.22.2 tomli-2.4.0 transformers-5.3.0 typer-0.24.1 tzdata-2025.3 xxhash-3.6.0 yarl-1.23.0
(cropper_env) es22btech11013@aicoe-a6000:~/divya/AFCIL/divya/cv-project/cropper$ cd data
(cropper_env) es22btech11013@aicoe-a6000:~/divya/AFCIL/divya/cv-project/cropper/data$ bash download.sh
==========================================
Cropper Dataset Download Script
==========================================

[1/3] GAICD Dataset
-------------------------------------------
Source: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch

To download GAICD dataset:

Option 1: Clone the repository and download from Google Drive
  git clone https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch.git
  # Download images from the Google Drive link in the repository README

Option 2: Direct download (if available)
  # The dataset contains 3,336 images with ~90 annotated crops each
  # - Training: 2,636 images
  # - Validation: 200 images
  # - Test: 500 images

Expected structure:
  GAICD/
  ├── images/
  │   ├── image_0001.jpg
  │   └── ...
  ├── annotations/
  │   ├── image_0001.txt  (contains crop coordinates and MOS scores)
  │   └── ...
  └── splits/
      ├── train.txt
      ├── val.txt
      └── test.txt

Cloning GAICD repository for reference...

[2/3] FCDB Dataset
-------------------------------------------
Source: https://github.com/yiling-chen/flickr-cropping-dataset

To download FCDB dataset:

Option 1: Clone the repository
  git clone https://github.com/yiling-chen/flickr-cropping-dataset.git

Option 2: Direct download
  # The dataset contains 348 test images
  # Each image has a single user-annotated crop box

Expected structure:
  FCDB/
  ├── images/
  │   ├── 1.jpg
  │   └── ...
  └── cropping_testing_set.json

Cloning FCDB repository...

[3/3] SACD Dataset
-------------------------------------------
Source: https://github.com/bcmi/Human-Centric-Image-Cropping

To download SACD dataset:

Option 1: Clone the repository and download from provided links
  git clone https://github.com/bcmi/Human-Centric-Image-Cropping.git
  # Follow the download instructions in the README

Option 2: Direct download
  # The dataset contains 2,906 images
  # - Training: 2,326 images
  # - Validation: 290 images
  # - Test: 290 images
  # Each image has multiple subject masks and corresponding ground-truth crops

Expected structure:
  SACD/
  ├── images/
  │   ├── image_001.jpg
  │   └── ...
  ├── masks/
  │   ├── image_001_mask_0.png
  │   └── ...
  ├── annotations/
  │   ├── image_001.json
  │   └── ...
  └── splits/
      ├── train.txt
      ├── val.txt
      └── test.txt

Cloning SACD repository...

==========================================
Download Script Complete
==========================================

Please ensure all datasets are placed in the correct directories:
  - GAICD: ./GAICD/
  - FCDB:  ./FCDB/
  - SACD:  ./SACD/

After downloading, verify the data with:
  python -c "from datasets import GAICDDataset; d = GAICDDataset('./GAICD', 'test'); print(f'GAICD test: {len(d)} images')"

(cropper_env) es22btech11013@aicoe-a6000:~/divya/AFCIL/divya/cv-project/cropper/data$ 
Cropper is a unified framework for various image cropping tasks:
- **Free-form cropping**: Identify visually appealing crops without constraints
- **Subject-aware cropping**: Crop images while preserving specific subjects (given by masks)
- **Aspect-ratio-aware cropping**: Generate crops with specified aspect ratios

The key innovations are:
1. **Visual prompt retrieval**: Automatically select relevant in-context examples using CLIP similarity
2. **Iterative refinement**: Progressively improve crop quality using scorer feedback

## Installation

### Create conda environment

```bash
conda create -n cropper python=3.10
conda activate cropper
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install additional dependencies for Mantis

```bash
pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
```

## Dataset Setup

### Download datasets

```bash
cd data
bash download.sh
```

This will provide instructions for downloading:
- **GAICD**: 3,336 images for free-form cropping
- **FCDB**: 348 images for aspect-ratio-aware cropping
- **SACD**: 2,906 images for subject-aware cropping

### Expected directory structure

```
data/
├── GAICD/
│   ├── images/
│   ├── annotations/
│   └── splits/
├── FCDB/
│   ├── images/
│   └── cropping_testing_set.json
└── SACD/
    ├── images/
    ├── masks/
    ├── annotations/
    └── splits/
```

## Quick Start

### Free-form cropping evaluation

```bash
python scripts/run_freeform.py \
    --data_dir ./data/GAICD \
    --config configs/default.yaml \
    --output_dir ./results/freeform \
    --device cuda
```

### Subject-aware cropping evaluation

```bash
python scripts/run_subject_aware.py \
    --data_dir ./data/SACD \
    --config configs/default.yaml \
    --output_dir ./results/subject_aware \
    --device cuda
```

### Aspect-ratio-aware cropping evaluation

```bash
python scripts/run_aspect_ratio.py \
    --data_dir ./data \
    --config configs/default.yaml \
    --output_dir ./results/aspect_ratio \
    --device cuda
```

### Run ablation studies

```bash
python scripts/ablation.py \
    --data_dir ./data \
    --output_dir ./results/ablation \
    --ablation all \
    --max_samples 50
```

## Expected Results

### Free-form cropping on GAICD (Table 9 from paper)

Using Mantis-8B-Idefics2:

| Metric | Paper | Expected |
|--------|-------|----------|
| Acc5   | 80.2  | ~80      |
| Acc10  | 88.6  | ~88      |
| SRCC   | 0.874 | ~0.87    |
| PCC    | 0.797 | ~0.80    |
| IoU    | 0.672 | ~0.67    |

### Subject-aware cropping on SACD (Table 3)

| Metric | Paper | Expected |
|--------|-------|----------|
| IoU    | 0.769 | ~0.77    |
| Disp   | 0.0372| ~0.04    |

### Aspect-ratio-aware cropping on FCDB (Table 4)

| Metric | Paper | Expected |
|--------|-------|----------|
| IoU    | 0.756 | ~0.75    |
| Disp   | 0.053 | ~0.05    |

## Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
# For Mantis-8B-Idefics2 (reduced from Gemini settings)
freeform:
  S: 10        # ICL examples (Gemini uses 30)
  T: 5         # GT crops per example
  R: 5         # Candidate crops (Gemini uses 6)
  L: 2         # Refinement iterations
  temperature: 0.05
  scorer: "vila+area"

subject_aware:
  S: 30
  L: 10        # More iterations for subject-aware
  scorer: "vila+area"

aspect_ratio:
  S: 10
  R: 6
  L: 2
  scorer: "clip"  # CLIP only for aspect-ratio
```

## Swapping VLMs

To use a different VLM, modify `vlm_model` in the config or pass a custom model:

```python
from cropper.models.vlm import create_vlm

# Use Idefics2 instead of Mantis
vlm = create_vlm(
    model_type="idefics2",
    model_name="HuggingFaceM4/idefics2-8b",
    device="cuda",
)
```

### Adding a new VLM

1. Create a new class inheriting from `BaseVLM` in `models/vlm.py`
2. Implement `generate()` and `parse_crops()` methods
3. Add to `create_vlm()` factory function

## Code Structure

```
cropper/
├── configs/
│   └── default.yaml          # Hyperparameters
├── data/
│   ├── download.sh           # Dataset download script
│   └── datasets.py           # Dataset loaders
├── models/
│   ├── vlm.py                # VLM wrapper (Mantis, Idefics2)
│   ├── clip_retriever.py     # CLIP-based retrieval
│   └── scorer.py             # VILA + CLIP + Area scorers
├── pipeline/
│   ├── prompt_builder.py     # VLM prompt construction
│   ├── retrieval.py          # ICL example retrieval
│   ├── refinement.py         # Iterative refinement
│   └── cropper.py            # Main pipeline
├── evaluation/
│   ├── metrics.py            # IoU, Disp, SRCC, PCC, AccK/N
│   └── evaluate.py           # Evaluation runner
├── utils/
│   ├── coord_utils.py        # Coordinate handling
│   └── visualization.py      # Visualization tools
└── scripts/
    ├── run_freeform.py       # Free-form evaluation
    ├── run_subject_aware.py  # Subject-aware evaluation
    ├── run_aspect_ratio.py   # Aspect-ratio evaluation
    └── ablation.py           # Ablation studies
```

## Known Limitations

1. **Context window**: Mantis-8B has a smaller context window than Gemini 1.5 Pro, limiting the number of ICL examples (S=10 vs S=30)

2. **Inference speed**: VLM inference is the bottleneck. Use checkpointing to resume interrupted runs.

3. **VILA-R scorer**: The original VILA-R model may be difficult to set up. Falls back to NIMA or heuristic scoring.

4. **Memory requirements**: Loading 10+ images into VLM context is memory-intensive. Use float16 and reduce image sizes if needed.

## Differences from Original Paper

1. Uses Mantis-8B-Idefics2 instead of Gemini 1.5 Pro
2. Reduced hyperparameters (S, R) to fit context window
3. Alternative aesthetic scorer (NIMA) if VILA-R unavailable
4. Open-source implementation for reproducibility

## Citation

```bibtex
@inproceedings{lee2025cropper,
  title={Cropper: Vision-Language Model for Image Cropping through In-Context Learning},
  author={Lee, Seung Hyun and Jiang, Jijun and Xu, Yiran and Li, Zhuofang and Ke, Junjie and Li, Yinxiao and He, Junfeng and Hickson, Steven and Datsenko, Katie and Kim, Sangpil and Yang, Ming-Hsuan and Essa, Irfan and Yang, Feng},
  booktitle={CVPR},
  year={2025}
}
```

## License

This is a research reimplementation. Please refer to the original paper and dataset licenses.
