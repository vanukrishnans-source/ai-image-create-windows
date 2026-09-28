"""How aic/resources/dml/unet_fp16.onnx was made (dev machine, ORT 1.24.4 + onnx 1.23): the models-v1 UNet graph with its
rebuilt unet.fp32.bin (plain copies, not links) converted with ONNX Runtime's float16 tool. Resize scale constants stay fp32.
The app never runs this: it re-creates unet.fp16.bin itself by casting unet.fp32.bin tensor-by-tensor with the same clamp
rules (aic/models.py derive_unet_fp16) and checks the SHA-256 recorded in unet_fp16.json.
usage: python make_unet_fp16.py path/to/unet.onnx unet"""
import onnx, numpy as np, onnxruntime as ort, time, sys
from onnxruntime.transformers.float16 import convert_float_to_float16
src, name = sys.argv[1], sys.argv[2]
m = onnx.load(src); t = time.time()
prod = {o: n for n in m.graph.node for o in n.output}
block = []
for n in m.graph.node:
    if n.op_type == "Resize":
        for i in n.input[1:]:
            if i in prod and prod[i].op_type == "Constant": block.append(prod[i].name)
print("blocked constants", len(block))
m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True, node_block_list=block)
print("converted", round(time.time() - t, 1))
onnx.save_model(m16, f"{name}.fp16.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location=f"{name}.fp16.bin", size_threshold=1024)
so = ort.SessionOptions(); so.log_severity_level = 3
try:
    s = ort.InferenceSession(f"{name}.fp16.onnx", so, providers=["CPUExecutionProvider"]); print("CPU session OK")
except Exception as e: print("CPU session FAILED:", str(e)[:500])
