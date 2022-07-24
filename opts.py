import argparse


parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument("dataset", type=str)
parser.add_argument("modality", type=str, choices=["RGB", "Flow"])
parser.add_argument("--train_list", type=str, default="")
parser.add_argument("--val_list", type=str, default="")
parser.add_argument("--root_path", type=str, default="")
parser.add_argument("--store_name", type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument("--arch", type=str, default="BNInception")
parser.add_argument("--num_segments", type=int, default=3)
parser.add_argument("--k", type=int, default=3)

parser.add_argument("--dropout", "--do", default=0.5, type=float, metavar="DO", help="dropout ratio (default: 0.5)")
parser.add_argument("--loss_type", type=str, default="nll", choices=["nll"])
parser.add_argument("--img_feature_dim", default=256, type=int, help="the feature dimension for each frame")
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--pretrain", type=str, default="imagenet")
parser.add_argument("--tune_from", type=str, default=None, help="fine-tune from checkpoint")
parser.add_argument("--syncbn", default=False, action="store_true")

# ========================= Learning Configs ==========================
parser.add_argument("--epochs", default=120, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 256)")
parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, metavar="LR", help="initial learning rate")
parser.add_argument("--lr_type", default="step", type=str, metavar="LRtype", help="learning rate type")
parser.add_argument(
    "--lr_steps",
    default=[50, 100],
    type=float,
    nargs="+",
    metavar="LRSteps",
    help="epochs to decay learning rate by 10",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay", "--wd", default=5e-4, type=float, metavar="W", help="weight decay (default: 5e-4)"
)
parser.add_argument(
    "--clip-gradient", "--gd", default=None, type=float, metavar="W", help="gradient norm clipping (default: disabled)"
)
parser.add_argument("--no_partialbn", "--npb", default=False, action="store_true")


# ========================= Monitor Configs ==========================
parser.add_argument("--print-freq", "-p", default=20, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--eval-freq", "-ef", default=5, type=int, metavar="N", help="evaluation frequency (default: 5)")


# ========================= Runtime Configs ==========================
parser.add_argument(
    "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
)
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--snapshot_pref", type=str, default="")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--gpus", nargs="+", type=int, default=None)
parser.add_argument("--flow_prefix", default="", type=str)
parser.add_argument("--root_log", type=str, default="log")
parser.add_argument("--root_model", type=str, default="checkpoint")
parser.add_argument("--dense_sample", default=False, action="store_true", help="use dense sample for video dataset")


# ========================= Few-Shot Configs =========================
parser.add_argument("--episodes", default=10000, type=int, help="number of episodes (default: 10,000)")
parser.add_argument("--way", default=5, type=int, help="few-shot way (default: 5)")
parser.add_argument("--shot", default=1, type=int, help="few-shot shot (default: 1)")
parser.add_argument("--n_query", default=1, type=int, help="few-shot shot (default: 1)")
parser.add_argument("--iter", default=50, type=int, help="number of prototypes update")
parser.add_argument("--seed", default=112, type=int, help="seed")
parser.add_argument("--transductive", default=False, action="store_true")
