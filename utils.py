import os
import shutil
from PIL import Image

import torch
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    """ImageFolder Dataset"""

    def __init__(self, root: str, transform=None) -> None:

        self.transform = transform
        self.samples = self.make_dataset(root)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        return image

    @staticmethod
    def make_dataset(directory):
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            for root, _, file_names in sorted(os.walk(target_dir, followlinks=True)):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    base, ext = os.path.splitext(path)
                    if ext.lower() in [".jpg", ".jpeg", ".png"]:
                        item = path, class_index
                        instances.append(item)

        return instances


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    max_k = max(top_k)
    batch_size = target.size(0)

    _, prediction = output.topk(max_k, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
