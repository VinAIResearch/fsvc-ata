import os
import random
import shutil

import numpy as np
import torch


try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision
from inference import inductive, transductive
from ops import dataset_config
from ops.ata import ATA
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.samplers import CategoriesSampler
from ops.transforms import GroupCenterCrop, GroupNormalize, GroupScale, IdentityTransform, Stack, ToTorchFormatTensor
from opts import parser
from train import train


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset, args.modality
    )
    full_arch_name = args.arch

    args.store_name = "_".join(
        [args.dataset, args.modality, full_arch_name, "segment%d" % args.num_segments, "e{}".format(args.epochs)]
    )
    if args.pretrain != "imagenet":
        args.store_name += "_{}".format(args.pretrain)
    if args.lr_type != "step":
        args.store_name += "_{}".format(args.lr_type)
    if args.dense_sample:
        args.store_name += "_dense"
    if args.suffix is not None:
        args.store_name += "_{}".format(args.suffix)
    print("storing name: " + args.store_name)

    check_rootfolders()

    model = TSN(
        num_class,
        args.num_segments,
        args.modality,
        base_model=args.arch,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        partial_bn=not args.no_partialbn,
        pretrain=args.pretrain,
        fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
    )

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    if "something" == args.dataset or "jester" == args.dataset:
        flip = False
    else:
        flip = True
    train_augmentation = model.get_augmentation(flip=flip, dataset=args.dataset)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, "cpu")
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint["epoch"])))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != "RGBDiff":
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == "RGB":
        data_length = 1
    elif args.modality in ["Flow", "RGBDiff"]:
        data_length = 5

    train_dataset = TSNDataSet(
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose(
            [
                train_augmentation,
                Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
                ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
                normalize,
            ]
        ),
        dense_sample=args.dense_sample,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = TSNDataSet(
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose(
            [
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
                ToTorchFormatTensor(div=(args.arch not in ["BNInception", "InceptionV3"])),
                normalize,
            ]
        ),
        dense_sample=args.dense_sample,
    )

    val_sampler = CategoriesSampler(val_dataset.label, args.episodes, args.way, args.shot + args.n_query)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=args.workers, batch_sampler=val_sampler, pin_memory=True
    )

    ata_helper = ATA(num_segments=args.num_segments)

    for group in policies:
        print(
            (
                "group: {} has {} params, lr_mult: {}, decay_mult: {}".format(
                    group["name"], len(group["params"]), group["lr_mult"], group["decay_mult"]
                )
            )
        )

    log = open(os.path.join(args.root_log, args.store_name, "log.csv"), "w")
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))

    if args.evaluate:
        if args.transductive:
            transductive(val_loader, model, ata_helper, args, log)
        else:
            inductive(val_loader, model, ata_helper, args, log)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, ata_helper, optimizer, epoch, num_class, args, log)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = inductive(val_loader, model, ata_helper, args, log)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            output_best = "Best Prec@1: %.3f\n" % (best_prec1)
            print(output_best)
            log.write(output_best + "\n")
            log.flush()
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_prec1": best_prec1,
                },
                is_best,
            )


def save_checkpoint(state, is_best):
    filename = "%s/%s/ckpt%s.pth.tar" % (args.root_model, args.store_name, str(state["epoch"]))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == "step":
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == "cos":
        import math

        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group["lr_mult"]
        param_group["weight_decay"] = decay * param_group["decay_mult"]


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log,
        args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print("creating folder " + folder)
            os.makedirs(folder)


if __name__ == "__main__":
    main()
