import time

import numpy as np
import torch
import torch.nn.functional as F
from ops.utils import AverageMeter, accuracy, get_one_hot
from tqdm import tqdm


def inductive(val_loader, model, ata_helper, args, log=None):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    target = torch.arange(args.way, dtype=torch.int16).repeat(args.shot)
    target = target.type(torch.LongTensor).cuda()
    target_1h = get_one_hot(target, args.way)
    target_1h = target_1h.cuda()

    support_idx, query_idx = (
        torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
        torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.n_query)))
        .long()
        .view(1, args.n_query, args.way),
    )

    for i, (input, _) in tqdm(enumerate(val_loader)):
        # compute output
        with torch.no_grad():
            outputepi = model(input, feature=True)

        support, query = outputepi[support_idx.flatten()].contiguous(), outputepi[query_idx.flatten()].contiguous()
        support = support.view(args.way * args.shot, args.num_segments, -1).detach()
        query = query.view(args.way * args.n_query, args.num_segments, -1).detach()

        support = F.normalize(support, dim=-1)
        query = F.normalize(query, dim=-1)

        weights = support.view(args.shot, args.way, args.num_segments, -1).mean(0).clone().detach().cuda()
        scale = torch.FloatTensor(1).fill_(10.0).cuda()

        weights.requires_grad_()
        scale.requires_grad_()
        optimizer = torch.optim.Adam([weights, scale])

        for step in range(args.iter):
            D, norm_D = ata_helper.similarity_matrix(support, weights, scale)

            sim_a = ata_helper.appearance_score(D, scale=1.0)

            ent_cond = -(norm_D * torch.log(norm_D + 1e-12)).sum(2).mean(1)
            ent = -(norm_D.mean(1) * torch.log(norm_D.mean(1) + 1e-12)).sum(1)

            loss = -(target_1h * torch.log(sim_a.softmax(-1) + 1e-12)).sum(1).mean(0)
            loss += (target_1h * ent_cond).sum(1).mean(0) * 0.1
            loss += -(target_1h * ent).sum(1).mean(0) * 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        D, norm_D = ata_helper.similarity_matrix(query, weights, scale)
        sim_a = ata_helper.appearance_score(D)
        sim_d = ata_helper.temporal_score(norm_D)

        sim = sim_a.softmax(1) + sim_d.softmax(1)

        # measure accuracy
        target = torch.arange(args.way, dtype=torch.int16).repeat(args.n_query).type(torch.LongTensor).cuda()

        (prec1,) = accuracy(sim.data, target.cuda(), topk=(1,))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = (
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(i, len(val_loader), batch_time=batch_time, top1=top1)
            )
            print(output)
            if log is not None:
                log.write(output + "\n")
                log.flush()

    output = "Testing Results: Prec@1 {top1.avg:.3f}".format(top1=top1)
    print(output)
    if log is not None:
        log.write(output + "\n")
        log.flush()
    return top1.avg


def transductive(val_loader, model, ata_helper, args, log=None):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    target = torch.arange(args.way, dtype=torch.int16).repeat(args.shot)
    target = target.type(torch.LongTensor).cuda()
    target_1h = get_one_hot(target, args.way)
    target_1h = target_1h.cuda()

    support_idx, query_idx = (
        torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
        torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.n_query)))
        .long()
        .view(1, args.n_query, args.way),
    )

    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(val_loader)):
            # compute output
            outputepi = model(input, feature=True)
            support, query = outputepi[support_idx.flatten()].contiguous(), outputepi[query_idx.flatten()].contiguous()
            support = support.view(args.way * args.shot, args.num_segments, -1).detach()
            query = query.view(args.way * args.n_query, args.num_segments, -1).detach()

            support = F.normalize(support, dim=-1)
            query = F.normalize(query, dim=-1)

            weights = support.view(args.shot, args.way, args.num_segments, -1).mean(0).clone().detach().cuda()
            scale = torch.FloatTensor(1).fill_(10.0).cuda()

            for step in range(args.iter):
                D, norm_D = ata_helper.similarity_matrix(query, weights, scale)

                sim_a = ata_helper.appearance_score(D)
                sim_d = ata_helper.temporal_score(norm_D)
                sim = sim_a.softmax(1) + sim_d.softmax(1)
                weights = (sim.unsqueeze(-1).unsqueeze(-1) * query.unsqueeze(1)).sum(0) / sim.unsqueeze(-1).unsqueeze(
                    -1
                ).sum(0) + support.view(args.shot, args.way, args.num_segments, -1).mean(0).clone().detach().cuda()
                weights = F.normalize(weights, dim=-1)

            D, norm_D = ata_helper.similarity_matrix(query, weights, scale)
            sim_a = ata_helper.appearance_score(D)
            sim_d = ata_helper.temporal_score(norm_D)

            sim = sim_a.softmax(1)  # + sim_d.softmax(1)

            # measure accuracy
            target = torch.arange(args.way, dtype=torch.int16).repeat(args.n_query).type(torch.LongTensor).cuda()

            (prec1,) = accuracy(sim.data, target.cuda(), topk=(1,))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, top1=top1
                    )
                )
                print(output)
                if log is not None:
                    log.write(output + "\n")
                    log.flush()

    output = "Testing Results: Prec@1 {top1.avg:.3f}".format(top1=top1)
    print(output)
    if log is not None:
        log.write(output + "\n")
        log.flush()
    return top1.avg
