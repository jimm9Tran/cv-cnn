import os
import time
import numpy as np
import PIL.Image
import torch
from torchvision import transforms as transforms

# Predefined metadata. Place Your Dataset Here Following the Key.

dataset_metalist = dict(
    fru92_full=dict(NUM_CLASSES=92,
                    DIR_TRAIN_IMAGES='/home/lcl/fru/fru92_lists/fru_train.txt',
                    DIR_TEST_IMAGES='/home/lcl/fru/fru92_lists/fru_test.txt',
                    IMAGE_PREFIX='/home/lcl/fru/fru92_images'),
    fru92_kfold=dict(
        NUM_CLASSES=92,
        DIR_TRAIN_IMAGES=
        '/home/lcl/fru/fru92_lists/fru92_fold/fru92_split_split_%d_train.txt',
        DIR_TEST_IMAGES=
        '/home/lcl/fru/fru92_lists/fru92_fold/fru92_split_split_%d_test.txt',
        IMAGE_PREFIX='/home/lcl/fru/fru92_images'),
    fru360_kfold=dict(
        NUM_CLASSES = 131,
        DIR_TRAIN_IMAGES=
        '/home/lcl/datasets/fruits-360_dataset/fruits-360/split_folds/fruit-360_split_%d_train.txt',
        DIR_TEST_IMAGES=
        '/home/lcl/datasets/fruits-360_dataset/fruits-360/split_folds/fruit-360_split_%d_test.txt',
        IMAGE_PREFIX='/home/lcl/datasets/fruits-360_dataset/fruits-360'),
    gro_kfold=dict(
        NUM_CLASSES = 27,
        DIR_TRAIN_IMAGES=
        '/home/lcl/datasets/fruits-360_dataset/fruits-360/split_folds/fruit-360_split_%d_train.txt',
        DIR_TEST_IMAGES=
        '/home/lcl/datasets/fruits-360_dataset/fruits-360/split_folds/fruit-360_split_%d_test.txt',
        IMAGE_PREFIX='/home/lcl/datasets/fruits-360_dataset/fruits-360'),
    veg81_kfold=dict(
        NUM_CLASSES = 125,
        DIR_TRAIN_IMAGES=
        '/home/lcl/datasets/fruitveg81/K_FOLD/fruitveg81_split_%d_train.txt',
        DIR_TEST_IMAGES=
        '/home/lcl/datasets/fruitveg81/K_FOLD/fruitveg81_split_%d_test.txt',
        IMAGE_PREFIX='/home/lcl/datasets/fruitveg81'),
    )

ImgLoader = lambda path: PIL.Image.open(path).convert('RGB')
print_freq = 100


class Empty_EMA():

    def __init__(self, decay):
        pass

    def register(self, model):
        pass

    def update(self, model):
        pass

    def apply_shadow(self, model):
        pass

    def restore(self, model):
        pass


class EMA():

    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, (name, self.shadow)
                new_average = (1.0 - self.decay
                               ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FruDataset(torch.utils.data.Dataset):

    def __init__(self,
                 txt_dir,
                 file_prefix,
                 transform=None,
                 target_transform=None,
                 loader=ImgLoader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((" ".join(words[:-1]), int(words[-1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.file_prefix = file_prefix

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = self.loader(os.path.join(self.file_prefix, img_name))
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_transform(mode, N, CROP, normalize):
    assert mode in ["train",
                    "test"], "specify your mode in `train` and `test`."
    pipeline = []
    if mode == "train":
        pipeline += [transforms.RandomHorizontalFlip(p=0.5)]
        crop_func = transforms.RandomCrop
    else:
        crop_func = transforms.CenterCrop
    pipeline += [
        transforms.Resize((N, N)),
        crop_func((CROP, CROP)),
        transforms.ToTensor(), normalize
    ]

    return transforms.Compose(pipeline)


def get_train_and_test_loader(metadata, k_fold, train_trans, test_trans,
                              train_batchsize, test_batchsize):
    """

    Args:
        metadata ([type]): [description]
        k_fold ([type]): 使用k折交叉验证的数据集，应在此处指定使用第几折。-1代表不适用交叉验证。
        train_trans ([type]): [description]
        test_trans ([type]): [description]
        train_batchsize ([type]): [description]
        test_batchsize ([type]): [description]
    """

    if k_fold >= 0:
        train_txt = metadata["DIR_TRAIN_IMAGES"] % (k_fold)
        test_txt = metadata["DIR_TEST_IMAGES"] % (k_fold)
    else:
        train_txt = metadata["DIR_TRAIN_IMAGES"]
        test_txt = metadata["DIR_TEST_IMAGES"]

    train_dataset = FruDataset(txt_dir=train_txt,
                               file_prefix=metadata["IMAGE_PREFIX"],
                               transform=train_trans)
    test_dataset = FruDataset(txt_dir=test_txt,
                              file_prefix=metadata["IMAGE_PREFIX"],
                              transform=test_trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batchsize,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batchsize,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1**(epoch // 40))
    if 'param_groups' in optimizer.state_dict():
        param_groups = optimizer.state_dict()['param_groups']
    else:
        param_groups = optimizer.module.state_dict()['param_groups']
    param_groups[0]['lr'] = lr
    param_groups[1]['lr'] = lr * 10


def train(train_loader,
          model,
          ema,
          criterion,
          optimizer,
          scheduler,
          epoch,
          accum=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / accum

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step

        loss.backward()

        if i % accum == 0:
            ema.update(model)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))
    # 最后一波 清空一下梯度 感觉没啥必要但写了试试（
    if i % accum != 0:
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model)
    # update scheduler
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    print("LR stepped to %.4f" % (lr))


def validate(val_loader, model, criterion, meta):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 添加对F1-score和Recall的支持
    # 我该怎么知道总类别有多少呢？
    # 用关键字传过来好了
    num = meta["NUM_CLASSES"]
    # 构建统计矩阵
    con_mat = np.zeros((num, num), dtype=np.int)

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            loss = torch.sum(loss)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))

            # 更新统计指标，用于之后其他指标（如召回率，F1 Score等）的计算
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()[0].cpu()
            target_var = target_var.cpu()
            for pred, pos in zip(pred, target_var):
                con_mat[pred, pos] += 1

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

    # 计算 Precision, Recall, F1-score等
    num_tot = np.sum(con_mat)
    num_tp = [con_mat[x, x] for x in range(num)]
    sum_tp = np.trace(con_mat)
    num_pred = np.sum(con_mat, axis=0)
    num_gts = np.sum(con_mat, axis=1)
    #print(num_gts)
    # 分类别得到统计结果
    cls_lv_prec = num_tp / num_pred
    cls_lv_recall = num_tp / num_gts
    # cls_lv_recall[np.isnan(cls_lv_recall)] = 0
    cls_lv_F1 = (2 * cls_lv_prec * cls_lv_recall) / (cls_lv_prec +
                                                     cls_lv_recall)

    micro_nametag = "micro"
    micro_prec = np.sum(cls_lv_prec * num_pred) / num_tot
    micro_recall = np.sum(num_tp) / num_tot
    # micro_F1 is never calculated because it generally equals to accuracy.
    weighted_F1 = np.sum(cls_lv_F1 * num_gts) / num_tot

    macro_nametag = "macro"
    macro_prec = np.mean(cls_lv_prec)
    macro_recall = np.mean(cls_lv_recall)
    macro_F1 = np.mean(cls_lv_F1)

    top1_acc = sum_tp / num_tot

    # 激光打印.jpg
    print("%10s|%10s|%10s|%10s" % ("", "Prec.", "Recall", "F1-score"))
    print(
        f"{macro_nametag:10s}|{macro_prec*100:10.3f}|{macro_recall*100:10.3f}|{macro_F1*100:10.3f}"
    )
    print(
        f"{micro_nametag:10s}|{micro_prec*100:10.3f}|{micro_recall*100:10.3f}|{weighted_F1*100:10.3f}"
    )
    print("(The above score is weighted F1-score.)")
    print(
        f"(For checking only) Top-1 Accu = micro F1-score = micro recall = micro precision = {top1_acc*100:.3f}\n"
    )

    # 打印Top-1/Top-5结果
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1,
                                                                  top5=top5))

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape((-1, )).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def deterministic_seed_torch(seed=1029):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别


def to_cuda(obj):
    device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    # multi-gpu condition
    if "," in device_ids:
        print(f"Multi-gpu mode applied. Transforming to GPUs:{device_ids}")
        device_ids = eval(f"[{device_ids}]")
        obj = torch.nn.DataParallel(obj, device_ids=device_ids)

    obj = obj.cuda()
    return obj


def to_distributed_devices(params, devices):
    """
    Put something/anything to the given devices.
    """
    params.to(torch.device("cuda", devices))
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        params = torch.nn.parallel.DistributedDataParallel(
            params,
            device_ids=[devices],
            output_device=devices,
            find_unused_parameters=True)
    return params