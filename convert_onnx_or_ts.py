import torch
import torch.onnx
import argparse
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from loguru import logger

from realesrgan import RealESRGANer  # adjust import based on your setup

# Try to use the modern torch.export-based ONNX exporter when available.
try:  # PyTorch >= 2.3
    from torch.export import Dim

    _HAS_TORCH_EXPORT = True
except Exception:  # pragma: no cover - older PyTorch versions
    _HAS_TORCH_EXPORT = False

def load_model(args):
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            logger.error(f"Model path {model_path} not found, please download the model from the internet.")
            exit(1)

    return model, model_path

def convert_onnx(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_name}.onnx")

    model_net, model_path = load_model(args)
    logger.info(f"Model loaded from {model_path}")

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # -----------------------------
    # 1. Load your model
    # -----------------------------
    device = "cuda"  # or "cpu"
    model = RealESRGANer(
        scale=4,          # adjust to your model
        model_path=model_path,
        dni_weight=dni_weight,
        device=device,
        model=model_net,
    ).model

    model.eval()  # set to evaluation mode

    # -----------------------------
    # 2. Prepare a dummy input
    # -----------------------------
    dummy_input = torch.randn(1, 3, 256, 256, device=device)  # 1x3xHxW

    # -----------------------------
    # 3. Define input/output names
    # -----------------------------

    # -----------------------------
    # 4. Dynamic axes (height and width)
    # -----------------------------
    # dynamic_axes = {
    #     "input": {2: "height", 3: "width"},
    #     "output": {2: "height", 3: "width"}
    # }

    dynamic_shapes = [(1, 3, Dim("height"), Dim("width"))]
    input_names = ["input"]
    output_names = ["output"]

    # -----------------------------
    # 5. Export to ONNX
    # -----------------------------
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,        # compatible with Video2X / NCNN
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        # dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,  # ensure eval mode
        dynamo=True,
        verbose=True,
    )

    print(f"ONNX model exported successfully to {output_path}")

def convert_ts(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model, model_path = load_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    if isinstance(model_path, list):
        # dni
        assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
        loadnet = dni(model_path[0], model_path[1], dni_weight)
    else:
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))

    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    # Use a modern ONNX opset to avoid failing version conversions (e.g. Resize -> older opsets).
    # Torch recommends opset_version >= 18 for the current exporter.
    ts_model = torch.jit.trace(model, dummy_input)
    output_path = os.path.join(output_dir, f"{args.model_name}.pt")
    ts_model.save(output_path)
    logger.info(f"Torchscript model exported to {output_path}")

def dni(net_a, net_b, dni_weight, key='params', loc='cpu'):
    """Deep network interpolation.

    ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
    """
    net_a = torch.load(net_a, map_location=torch.device(loc))
    net_b = torch.load(net_b, map_location=torch.device(loc))
    for k, v_a in net_a[key].items():
        net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
    return net_a

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument("-o", "--output_dir", type=str, default="onnx_or_ts_models")
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--convert_to_ts', action='store_true', default=False, help='Convert to torchscript model')

    args = parser.parse_args()
    if args.convert_to_ts:
        convert_ts(args)
    else:
        convert_onnx(args)