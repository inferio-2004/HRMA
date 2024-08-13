import argparse
import os
from glob import glob
import shutil
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import archs
from dataset import Dataset
from metrics import iou_score, ppv, sensitivity, hausdorff_distance,asd
from utils import AverageMeter
import numpy as np
import torch.nn as nn
from config import get_ClusterSeg_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="CANS2", help='model name')
    parser.add_argument('--dataset', default='LiTS', help='dataset name')
    parser.add_argument('--data_dir', default='/kaggle/input/lits-dataset', help='path to dataset directory')
    parser.add_argument('--img_ext', default='.png', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')
    parser.add_argument('--suffix', default='_woDS', help='model suffix')

    return parser.parse_args()


    return args


def main():
    args = parse_args()
    dataset_model = args.dataset + args.name + args.suffix
    network_name = args.name

    # Network version control code (unchanged)
    network_version = None
    nvc = os.path.join('./network_version_controll/')
    list_nvc = os.listdir(nvc)
    for dir_name in list_nvc:
        if os.path.isdir(os.path.join(nvc, dir_name)):
            if network_name == dir_name.split('_')[-1]:
                network_version = dir_name
                break

    # Copy network files
    path = os.path.join('./network_version_controll/', network_version)
    is_file = True
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)):
            if filename[0] == '.':
                continue
            is_file = False
            path = os.path.join(path, filename)
            break

    if is_file:
        origin_path = os.path.join('./network_version_controll/', network_version, args.name + '.py')
        target_path = os.path.join('./Modules/General_Network.py')
        shutil.copyfile(origin_path, target_path)
    else:
        for filename in os.listdir(path):
            origin_path = os.path.join(path, filename)
            if filename.split('.')[0] == args.name:
                target_path = os.path.join('./Modules/General_Network.py')
            else:
                target_path = os.path.join('./Modules/', filename)
            shutil.copyfile(origin_path, target_path)

    # Load config
    with open('models/%s/config.yml' % dataset_model, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True
    from Modules.General_Network import get_network
    print("=> creating model %s" % config['arch'])

    model_val = get_network()
    if isinstance(model_val, nn.Module):
        model = model_val
    else:
        model = model_val()
    model = model.cuda()

    # Data loading code for PNG files
    img_ids = glob(os.path.join(args.data_dir, 'test_vol_h5', '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    imgs = [np.array(Image.open(os.path.join(args.data_dir, 'test_vol_h5', img_id + args.img_ext)).convert('L')) for img_id in img_ids]
    masks = [np.array(Image.open(os.path.join(args.data_dir, 'test_vol_h5', img_id + args.mask_ext)).convert('L')) for img_id in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.9, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test_vol_h5'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test_vol_h5_label'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    ppv_avg_meter = AverageMeter()
    sensitivity_avg_meter = AverageMeter()
    hausdorff_distance_avg_meter = AverageMeter()
    pixel_accuracy_avg=AverageMeter()
    asd_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice = iou_score(output, target)
            pa=pixel_accuracy(ouput,target)
            pp_v = ppv(output, target)
            sen_sitivity = sensitivity(output, target)

            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            ppv_avg_meter.update(pp_v, input.size(0))
            pixel_accuracy_avg.update(pa,input.size(0))
            sensitivity_avg_meter.update(sen_sitivity, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            img_out = output.copy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            hausdorffdistance = hausdorff_distance(output.astype(np.uint8),
                                                   target.cpu().numpy().astype(np.uint8))
            asd_ = asd(output.astype(np.uint8),target.cpu().numpy().astype(np.uint8))
            hausdorff_distance_avg_meter.update(hausdorffdistance, input.size(0))
            asd_avg_meter.update(asd_,input.size(0))
            from PIL import Image
            import matplotlib.pyplot as plt

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    # gray_array = img_out[i, c] * 255
                    gray_array = img_out[i, c] * 255
                    # plt.imshow(gray_array, cmap='viridis')
                    # plt.imshow(gray_array, cmap='viridis')
                    # plt.show(gray_array)
                    plt.imsave(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),gray_array)
                    # cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                    #             ((gray_array)))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('ppv: %.4f' % ppv_avg_meter.avg)
    print('sensitivity: %.4f' % sensitivity_avg_meter.avg)
    print('hausdorffdistance: %.4f' % hausdorff_distance_avg_meter.avg)
    print('avg: %.4f' % asd_avg_meter.avg)
    print('pa: %.4f' % pixel_accuracy_avg.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
