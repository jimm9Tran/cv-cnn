import argparse

import shared_lib
import torch
import torch.nn as nn
import torch.optim as optim
from cbam_model import MODEL_CBAM
from shared_lib import (adjust_learning_rate, dataset_metalist,
                        deterministic_seed_torch, get_train_and_test_loader,
                        get_transform, train, validate, EMA, Empty_EMA)
from simplified_main_model import MODEL, SENet_BASE
from torchvision import transforms as transforms

deterministic_seed_torch(seed=42)


# ==================================================================
# Constants
# ==================================================================
EPOCH = 200  # number of times for each run-through
EXPECT_BATCH_SIZE = 64  # number of images for each iteration. For gredient Accumulation
BATCH_SIZE = 16  # actual num of images per GPU.
LEARNING_RATE = 1e-2  # default learning rate. 已等比放大至64
N = 256  # size of input images (512 or 640)
CROP = 224
port = 11451
MODEL_ZOO = dict(MODEL=MODEL, MODEL_CBAM=MODEL_CBAM, MODEL_BASE=SENet_BASE)
GPU_IN_USE = torch.cuda.is_available()  # whether using GPU
PATH_MODEL_PARAMS = None
INIT_WEIGHT_PATH = '/home/lcl/fru/senet154-c7b49a05.pth'

# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(
    description='Pytorch Implementation of Nasnet Finetune')
parser.add_argument('--lr',
                    default=LEARNING_RATE,
                    type=float,
                    help='learning rate')
parser.add_argument('--epoch',
                    default=EPOCH,
                    type=int,
                    help='number of epochs')
parser.add_argument('--model', default=None, type=str, help="applying model")
parser.add_argument('--dataset',
                    default=None,
                    type=str,
                    help="train/test on which dataset?")
parser.add_argument('--kfold', default=None, type=int, help="use which fold?")
parser.add_argument('--mode',
                    default=None,
                    type=str,
                    help="train or test only?")
parser.add_argument('--trainBatchSize',
                    default=BATCH_SIZE,
                    type=int,
                    help='training batch size')
parser.add_argument('--testBatchSize',
                    default=BATCH_SIZE,
                    type=int,
                    help='testing batch size')
parser.add_argument('--pathModelParams',
                    default=PATH_MODEL_PARAMS,
                    type=str,
                    help='path of model parameters')
parser.add_argument('--weightpath',
                    default=INIT_WEIGHT_PATH,
                    type=str,
                    help='inint weight path')
parser.add_argument('--no-ema', action="store_true", help='do not apply EMA.')
parser.add_argument('--print_freq', default=50, type=int, help='print')

args = parser.parse_args()
# set print_freq
shared_lib.print_freq = args.print_freq
# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')

dataset_key = args.dataset
meta_info = dataset_metalist[dataset_key]
kfold = args.kfold

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transforms = get_transform("train", N, CROP, normalize)
test_transforms = get_transform("test", N, CROP, normalize)
train_loader, test_loader = get_train_and_test_loader(
    meta_info,
    k_fold=kfold,
    train_trans=train_transforms,
    test_trans=test_transforms,
    train_batchsize=args.trainBatchSize,
    test_batchsize=args.testBatchSize)

print('Done.')

# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

# INIT TRAINING MODEL
assert args.model in ("MODEL", "MODEL_CBAM", "MODEL_BASE")
main_model = MODEL_ZOO[args.model](num_classes=meta_info['NUM_CLASSES'],
                                   senet154_weight=INIT_WEIGHT_PATH,
                                   multi_scale=True)
criterion = nn.CrossEntropyLoss()
ema = Empty_EMA(None)

if args.mode == "test":
    main_model.load_state_dict(torch.load(args.pathModelParams))
elif args.mode == 'train':
    # 构造最新权重名称
    latest_dict_fname = ".".join(
        args.pathModelParams.split(".")[:-1]) + "_lastest.pth"
    # 构造日志名称
    log_filename = ".".join(args.pathModelParams.split(".")[:-1]) + "_log.txt"

    ignored_params = list(map(
        id, main_model.global_out.parameters()))  #layer need to be trained
    base_params = filter(lambda p: id(p) not in ignored_params,
                         main_model.parameters())
    optimizer = optim.SGD([{
        'params': base_params
    }, {
        'params': main_model.global_out.parameters(),
        'lr': args.lr * 10
    }],
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.00001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                     T_max=args.epoch)
    
    if not args.no_ema:
        ema = EMA(decay=0.999)

assert EXPECT_BATCH_SIZE % BATCH_SIZE == 0, "Gradient Accumulation requires <int> iteration!"
accum_iter = EXPECT_BATCH_SIZE // BATCH_SIZE
if accum_iter > 1:
    print(f"Apply Gradient Accumulation. Eq Batchsize={EXPECT_BATCH_SIZE}.")

if GPU_IN_USE:
    #cudnn.benchmark = True
    main_model = main_model.cuda()
    #optimizer = optimizer.cuda()
    criterion = criterion.cuda()
    ema.register(main_model)

if args.mode == "train":
    best_prec1 = 0
    for epoch in range(0, args.epoch):
        #adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader,
              main_model,
              ema,
              criterion,
              optimizer,
              scheduler,
              epoch,
              accum=accum_iter)

        # evaluate on validation set
        ema.apply_shadow(main_model)
        prec1, prec5 = validate(test_loader, main_model, criterion, meta_info)
        # prec1 = test(epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if prec1 > best_prec1:
            torch.save(main_model.state_dict(), args.pathModelParams)
            print('Checkpoint saved to {}'.format(args.pathModelParams))

        #  每一轮保存最新模型。
        torch.save(main_model.state_dict(), latest_dict_fname)
        print('Save the lastest model to {}'.format(args.pathModelParams))

        # 还原EMA
        ema.restore(main_model)

        #  写日志
        with open(log_filename, mode='a', encoding="utf-8") as f:
            f.write(
                f"Epoch[{epoch}/{args.epoch}]: Prec@1 {prec1:.3f}; Prec@5 {prec5:.3f}\n"
            )
        best_prec1 = max(prec1, best_prec1)

    print("Training Finished. Best Prec1: ", best_prec1)

elif args.mode == "test":
    prec1 = validate(test_loader, main_model, criterion, meta_info)
