#!/usr/bin/env bash

# 0.4.x 之后的版本需要 bokeh > 2.
# nvdashboard 0.3.1 可能与 jupyter lab 版本不兼容
# 参照 NVIDIA 发布的 CLARA 视频确定相关的版本, 均为 0.2.1

pip install jupyterlab-nvdashboard==0.2.1
jupyter labextension install jupyterlab-nvdashboard@0.2.1

echo ------------------
echo ------jupyterlab intallation completed
echo ------------------
echo -- fix bokeh issue downgrad to 1.4.0
pip uninstall -y bokeh
pip install bokeh==1.4.0
echo ------------------ bokeh installed
nvidia-smi
echo ------------------

jupyter lab /workspace --ip 0.0.0.0 --port 19010 --allow-root --no-browser --NotebookApp.token=""
