ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ADD $WHL_FILE /tmp/$WHL_FILE
RUN /opt/conda310/bin/pip install /tmp/$WHL_FILE \
    -i https://artifacts.antgroup-inc.cn/simple/ \
    --extra-index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url=https://download.pytorch.org/whl/cu126 \
    && rm /tmp/$WHL_FILE

RUN if /opt/conda310/bin/python -c "import importlib.metadata as md, sys; reqs = md.requires('rtp_llm') or []; sys.exit(0 if any(req.lower().replace('_', '-').startswith('torch-memory-saver') for req in reqs) else 1)"; then \
        /opt/conda310/bin/python -c "import pathlib, torch_memory_saver; site = pathlib.Path(torch_memory_saver.__file__).resolve().parent.parent; candidates = sorted(site.glob('torch_memory_saver_hook_mode_preload_cu12*.so')) or sorted(site.glob('torch_memory_saver_hook_mode_preload*.so')); assert candidates, 'torch_memory_saver preload hook so not found'; print(candidates[0])" > /tmp/torch_memory_saver_preload_so \
        && LD_PRELOAD="$(cat /tmp/torch_memory_saver_preload_so)" /opt/conda310/bin/python -c "import ctypes; root = ctypes.CDLL(None); assert hasattr(root, 'tms_pause'), 'tms_pause not exported by torch_memory_saver preload hook'; assert hasattr(root, 'tms_resume'), 'tms_resume not exported by torch_memory_saver preload hook'" \
        && rm /tmp/torch_memory_saver_preload_so; \
    else \
        echo "rtp_llm wheel does not require torch_memory_saver; skip preload hook validation"; \
    fi

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
