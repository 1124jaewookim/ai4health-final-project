import argparse

def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--batch_size", type=int, default=256, help="_")
    parser.add_argument("--epochs_num", type=int, default=1, help="_")
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=20234010, help="_")

    args = parser.parse_args()
    return args