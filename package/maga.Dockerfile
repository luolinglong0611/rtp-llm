ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ADD $WHL_FILE /tmp/$WHL_FILE
RUN /opt/conda310/bin/pip install /tmp/$WHL_FILE \
    -i https://artifacts.antgroup-inc.cn/simple/ \
    --extra-index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url=https://download.pytorch.org/whl/cu126 \
    && rm /tmp/$WHL_FILE

RUN /opt/conda310/bin/python - <<'PY'
import importlib.metadata as md
import importlib.util
import pathlib

reqs = md.requires("rtp_llm") or []
needs_tms = any(
    req.lower().replace("_", "-").startswith("torch-memory-saver")
    for req in reqs
)
if not needs_tms:
    print("rtp_llm wheel does not require torch_memory_saver; skip preload hook validation")
    raise SystemExit(0)

spec = importlib.util.find_spec("torch_memory_saver")
assert spec and spec.origin, "torch_memory_saver package not found"
site = pathlib.Path(spec.origin).resolve().parent.parent
candidates = (
    sorted(site.glob("torch_memory_saver_hook_mode_preload_cu12*.so"))
    or sorted(site.glob("torch_memory_saver_hook_mode_preload*.so"))
)
assert candidates, "torch_memory_saver preload hook so not found"
so_path = candidates[0]
data = so_path.read_bytes()
for symbol in (b"tms_pause\x00", b"tms_resume\x00"):
    assert symbol in data, f"{symbol[:-1].decode()} not found in torch_memory_saver preload hook"
print(so_path)
PY

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
