import argparse
import torch
from torch import nn
from glow_wb import Camera_Glow_norev_re


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
    parser.add_argument('--n_flow', type=int, default=8)
    parser.add_argument('--n_block', type=int, default=2)
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--no_lu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu')
    model = Camera_Glow_norev_re(3, 15, args.n_flow, args.n_block,
                                 affine=args.affine, conv_lu=not args.no_lu)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    wrapper = WBFlowWrapper(model)

    img = torch.randn(1, 3, 256, 256)
    camidx = torch.zeros(1, 15, 1, 1)
    camidx[:, 0] = 1.0

    torch.onnx.export(
        wrapper,
        (img, camidx),
        args.output,
        opset_version=args.opset,
        input_names=['input', 'camidx'],
        output_names=['output'],
    )


if __name__ == '__main__':
    main()
