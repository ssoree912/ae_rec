import argparse


def build_rectified_parser():
    parser = argparse.ArgumentParser(description="Base options for SR-Rectified pipeline")
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--sr_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_channels", type=int, default=32)
    return parser

