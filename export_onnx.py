import argparse
import torch
from glow_wb import Camera_Glow_norev_re
from torch import nn

class WBFlowWrapper(nn.Module):
    def __init__(self, glow_model):
        super().__init__()
        self.glow = glow_model

    def forward(self, img, camidx):
        z = self.glow(img, camidx, forward=True)
        out = self.glow(z, camidx, forward=False)
        return out

def main():
    parser = argparse.ArgumentParser(description="Export WBFlow model to ONNX")
    parser.add_argument('--weights', required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--output', default='wbflow.onnx', help='Output ONNX file')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()

    device = torch.device('cpu')
    model = Camera_Glow_norev_re(3, 15, n_flow=8, n_block=2, affine=False, conv_lu=True)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    wrapped = WBFlowWrapper(model)

    dummy_img = torch.randn(1, 3, 256, 256, device=device)
    dummy_cam = torch.zeros(1, 15, 1, 1, device=device)
    torch.onnx.export(
        wrapped,
        (dummy_img, dummy_cam),
        args.output,
        input_names=['input', 'camidx'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'camidx': {0: 'batch'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=args.opset,
    )
    print(f"ONNX model saved to {args.output}")

if __name__ == '__main__':
    main()
