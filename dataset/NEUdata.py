import cv2
from sklearn.model_selection import KFold
import os
import yaml
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import SimpleITK as sitk
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

with open("argument/test_config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

test_dataset = config["dataset"]
val_dataset = config["val_ds"]


def tensor2numpy(input_tensor: torch.Tensor):
    input_tensor = input_tensor.squeeze().to(torch.device('cpu')).numpy()
    # in_arr = numpy.transpose(input_tensor, (
    #     1, 2, 0))  # 将(c,w,h)转换为(w,h,c)。但此时如果是全精度模型，转化出来的dtype=float64 范围是[0,1]。后续要转换为cv2对象，需要乘以255
    cvimg = (input_tensor * 255).astype(np.uint8)
    # cvimg = cv2.cvtColor(np.uint8(in_arr * 255), cv2.COLOR_GRAY2BGR)
    return cvimg


def save_img_cv2(res, name, type):
    res = tensor2numpy(res.squeeze().permute(1, 2, 0))
    # res = Image.fromarray(res*255)
    # res = res.permute(1,2,0).numpy().squeeze()
    path = rf"E:\hr\work\Diff-UNet-main\BraTS2020\test_results\temp\{type}_{name}.png"
    # os.makedirs(path, exist_ok=True)
    # res.save(path)
    # cv2.imwrite(path, res*255)
    cv2.imwrite(path, res)


def cv_random_flip(img, label, depth, edge):
    # flip_flag = random.randint(0, 1)
    flip_flag = 1
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        # label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if label is not None:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            label = None
            # edge = None

        if edge is not None:
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # label = None
            edge = None
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    #     if label is not None:
    #         label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     else:
    #         label = None
    #     if edge is not None:
    #         edge = edge.transpose(Image.FLIP_TOP_BOTTOM)
    #     else:
    #         edge = None

    return img, label, depth, edge


def randomCrop(image, label, depth, edge):
    border = 10
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    # if random_region[3] < random_region[1]:
    #     temp = random_region[3]
    #     random_region[3] = random_region[1]
    #     random_region[1] = temp
    if label is not None:
        if edge is not None:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(
                random_region)
        else:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge
    else:
        return image.crop(random_region), label, depth.crop(random_region), edge


def randomCrop_unlabel(image, label, depth, edge):
    border = 60
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    if label is not None:
        if edge is not None:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(
                random_region)
        else:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge
    else:
        return image.crop(random_region), label, depth.crop(random_region), edge


def randomRotation(image, label, depth, edge):
    mode = Image.BICUBIC
    if random.random() > 0:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        if label is not None:
            label = label.rotate(random_angle, mode)
        if edge is not None:
            edge = edge.rotate(random_angle, mode)
    return image, label, depth, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def normalize(img, depth=None, mask=None, trainsize=256):
    img = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if depth is not None:
        depth = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(depth)
    if mask is not None:
        # mask = torch.from_numpy(np.array(mask))
        mask = transforms.Compose([transforms.Resize((trainsize, trainsize)), transforms.ToTensor()])(mask)
        # x = np.array(mask)
        return img, depth, mask.long()
    return img, depth


def resize(img, depth, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))
    # print('x', w, h ,long_side)
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    depth = depth.resize((ow, oh), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((ow, oh), Image.NEAREST)
    return img, depth, mask


def crop(img, depth, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    # masks =
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    depth = depth.crop((x, y, x + size, y + size))
    if mask is not None:
        mask = mask.crop((x, y, x + size, y + size))

    return img, depth, mask


def hflip(img, depth, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def resample_img(
        image: sitk.Image,
        out_spacing=(2.0, 2.0, 2.0),
        out_size=None,
        is_label: bool = False,
        pad_value=0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


class PretrainDataset(Dataset):
    def __init__(self, test_flag='Train'):

        self.test_flag = test_flag

        if test_flag == 'Train':
            image_root = 'rgb'
            depth_root = 'd'
            gt_root = 'gt'
            sp_root = 'gt-sp'
        elif test_flag == 'Val':
            image_root = 'rgb-test'
            depth_root = 'd-test'
            gt_root = 'gt-test'
            sp_root = 'gt-sp'
        elif test_flag == 'Test':
            image_root = 'rgb-test'
            depth_root = 'd-test'
            gt_root = 'gt-test'
            sp_root = 'gt-sp'

        self.trainsize = 352
        self.images = [image_root + '\\' + f for f in os.listdir(image_root) if f.endswith('.bmp')]
        self.depths = [depth_root + '\\' + f for f in os.listdir(depth_root) if
                       f.endswith('.tiff') or f.endswith('.jpg')]
        self.gts = [gt_root + '\\' + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.sps = [sp_root + '\\' + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.sps = sorted(self.sps)
        self.depths = sorted(self.depths)

        print(f'{test_flag} : images({len(self.images)}) , depths({len(self.depths)}) , edge({len(self.edges)})')
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.tensor_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.gt_loader(self.gts[index])
        sp = self.gt_loader(self.sps[index])
        name = self.images[index].split('/')[-1]
        if self.test_flag == 'Test':
            gt = self.tensor_transform(gt)
            image = self.img_transform(image)
            depth = self.depth_transform(depth)
            image_rgbt = torch.cat((image, depth), dim=0)

            return image_rgbt, gt, name
        elif self.test_flag == 'Val':
            gt = self.tensor_transform(gt)
            image = self.img_transform(image)
            depth = self.depth_transform(depth)
            image_rgbt = torch.cat((image, depth), dim=0)

            return image_rgbt, gt
        else:
            gt = self.gt_transform(gt)
            sp = self.gt_transform(sp)
            image = self.edge_transform(image)
            image = self.normal(image)
            depth = self.depth_transform(depth)
            image_rgbt = torch.cat((image, depth), dim=0)

            return image_rgbt, gt, sp

    def __len__(self):
        return len(self.images)

    def filter_files(self, test_flag):
        assert len(self.images) == len(self.gts)
        print('edge=', len(self.edges), 'gts=', len(self.gts))
        images = []
        depths = []
        gts = []
        edges = []

        i = 0
        data_size = len(self.images)
        if test_flag == 'Val':
            data_size = 357
        for img_path, depth_path, gt_path, edge_path in zip(self.images, self.depths, self.gts, self.edges):
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            i += 1
            if img.size == gt.size and img.size == edge.size and i <= data_size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.depths = depths
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return Image.fromarray(img, "L")

    def gt_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt


def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res


class Args:
    def __init__(self) -> None:
        self.workers = 8
        self.fold = 0
        self.batch_size = 2


def get_loader_brats():
    train_ds = PretrainDataset(test_flag='Train')

    val_ds = PretrainDataset(test_flag='Val')

    test_ds = PretrainDataset(test_flag='Test')

    print(f"train is {len(train_ds)}, val is {len(val_ds)}, test is {len(test_ds)}")

    loader = [train_ds, val_ds, test_ds]

    return loader
