import time

import torch
from ops.utils import AverageMeter, accuracy, get_one_hot
from tqdm import tqdm


def train(train_loader, model, ata_helper, optimizer, epoch, num_class, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in tqdm(enumerate(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        target_one_hot = get_one_hot(target, num_class)
        target_one_hot = target_one_hot.cuda()
        target = target.cuda()

        # compute output
        D, norm_D = model(input)

        sim_a = ata_helper.appearance_score(D, scale=1.0)
        sim_d = ata_helper.temporal_score(norm_D, scale=1.0)

        ent_cond = -(norm_D * torch.log(norm_D + 1e-12)).sum(2).mean(1)
        ent = -(norm_D.mean(1) * torch.log(norm_D.mean(1) + 1e-12)).sum(1)

        loss = -(target_one_hot * torch.log(sim_a.softmax(-1) + 1e-12)).sum(1).mean(0)
        loss += (target_one_hot * ent_cond).sum(1).mean(0) * 0.1
        loss += -(target_one_hot * ent).sum(1).mean(0) * 1
        loss += -(target_one_hot * sim_d).sum(1).mean(0) * 0.05

        sim = sim_a.softmax(1) + sim_d.softmax(1)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(sim.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
        if i % args.print_freq == 0:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t"
                "Mem {mem:}".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"] * 0.1,
                    mem=mem,
                )
            )  # TODO
            print(output)
            log.write(output + "\n")
            log.flush()
