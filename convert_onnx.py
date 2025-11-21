# export_and_prepare_onnx.py
import torch
import onnx
import argparse
import subprocess
from basicsr.archs.rrdbnet_arch import RRDBNet
import os

def load_checkpoint(pth):
    ckpt = torch.load(pth, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'params_ema' in ckpt:
            return ckpt['params_ema']
        if 'params' in ckpt:
            return ckpt['params']
    return ckpt

def export_onnx(pth, out_onnx, blocks=23, nf=64, gc=32, scale=4, size=256, opset=11):
    state = load_checkpoint(pth)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=nf,
                    num_block=blocks, num_grow_ch=gc, scale=scale)
    # try direct load, then fallback to prefix-stripping
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        new_state = {}
        for k, v in state.items():
            new_k = k
            if k.startswith('params.'):
                new_k = k[len('params.'):]
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            new_state[new_k] = v
        model.load_state_dict(new_state, strict=True)

    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        out_onnx,
        input_names=['data'],
        output_names=['output'],
        # dynamic_axes={'data': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}},
        opset_version=opset,
        do_constant_folding=True
    )
    print("Exported ONNX:", out_onnx)

def rename_onnx_io(src_onnx, dst_onnx, new_in='data', new_out='output'):
    model = onnx.load(src_onnx)
    # assume single input / single output
    if len(model.graph.input) == 1:
        model.graph.input[0].name = new_in
    if len(model.graph.output) == 1:
        model.graph.output[0].name = new_out
    # rename value_info and node input/output references
    def rename_all(name_from, name_to):
        for v in model.graph.value_info:
            if v.name == name_from:
                v.name = name_to
        for n in model.graph.node:
            for i, iname in enumerate(n.input):
                if iname == name_from:
                    n.input[i] = name_to
            for i, oname in enumerate(n.output):
                if oname == name_from:
                    n.output[i] = name_to
    rename_all('in0', new_in)
    rename_all('out0', new_out)
    onnx.save(model, dst_onnx)
    print("Renamed ONNX IO to:", new_in, new_out, "->", dst_onnx)

def simplify_onnx(onnx_path, out_path):
    try:
        import onnxsim
    except Exception:
        print("onnx-simplifier not installed; skipping simplification.")
        return False
    import onnx
    model = onnx.load(onnx_path)
    model_simp, check = onnxsim.simplify(model)
    if not check:
        raise RuntimeError("onnxsim failed to validate simplified model")
    onnx.save(model_simp, out_path)
    print("Simplified ONNX saved:", out_path)
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--onnx_renamed", default=None)
    p.add_argument("--blocks", type=int, default=23)
    p.add_argument("--nf", type=int, default=64)
    p.add_argument("--gc", type=int, default=32)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    export_onnx(args.pth, args.onnx, args.blocks, args.nf, args.gc, args.scale, args.size)

    # optionally rename in0/out0 -> data/output (matching your working param)
    if args.onnx_renamed:
        rename_onnx_io(args.onnx, args.onnx_renamed, new_in='data', new_out='output')
        onnx_to_simplify = args.onnx_renamed
    else:
        onnx_to_simplify = args.onnx

    # simplify if possible to reduce strange ops
    simp_out = os.path.splitext(onnx_to_simplify)[0] + "-sim.onnx"
    try:
        if simplify_onnx(onnx_to_simplify, simp_out):
            print("Using simplified ONNX:", simp_out)
            print("Now run: onnx2ncnn", simp_out, "model.param model.bin")
    except Exception as e:
        print("onnx simplifier failed:", e)
        print("Proceed with raw ONNX:", onnx_to_simplify)
