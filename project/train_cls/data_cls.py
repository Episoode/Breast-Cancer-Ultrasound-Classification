import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CutMix:
    '''
    功能说明:
        对超声图像采用CutMix增强
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img1, img2, bbox1, bbox2):
        if random.random() < self.prob:
            return self.apply_cutmix_aug(img1, img2, bbox1, bbox2)
        else:
            return img1

    def apply_cutmix_aug(self, img1, img2, bbox1, bbox2):
        w1, h1 = img1.size
        w2, h2 = img2.size

        cx1, cy1, bw1, bh1 = bbox1
        cx2, cy2, bw2, bh2 = bbox2

        # 计算图片1的结点区域坐标 (左上和右下坐标)
        x1_min = max(int(cx1 * w1 - bw1 * w1 / 2), 0)
        y1_min = max(int(cy1 * h1 - bh1 * h1 / 2), 0)
        x1_max = min(int(cx1 * w1 + bw1 * w1 / 2), w1)
        y1_max = min(int(cy1 * h1 + bh1 * h1 / 2), h1)

        # 裁剪图片1的结点区域
        img1_cropped = img1.crop((x1_min, y1_min, x1_max, y1_max))

        # 检查是否需要放大图片1的结点区域
        if bw1 < bw2 or bh1 < bh2:
            img1_cropped_resized = img1_cropped.resize((int(bw2 * w2), int(bh2 * h2)), Image.BILINEAR)
        else:
            # 1的结点区域的宽高不超过图片2结点宽高的两倍
            max_width = int(bw2 * w2 * 2)
            max_height = int(bh2 * h2 * 2)
            img1_cropped_resized = img1_cropped.resize(
                (min(img1_cropped.width, max_width), min(img1_cropped.height, max_height)), Image.BILINEAR)

        # 计算图片2中结点的中心坐标
        cx2, cy2 = int(cx2 * w2), int(cy2 * h2)

        # 使用图片1的结点区域的宽度和高度来计算在图片2中的粘贴位置
        bw1_resized, bh1_resized = img1_cropped_resized.size
        x2_min = max(int(cx2 - bw1_resized / 2), 0)
        y2_min = max(int(cy2 - bh1_resized / 2), 0)

        #合成
        img2_pasted = img2.copy()
        img2_pasted.paste(img1_cropped_resized, (x2_min, y2_min))

        return img2_pasted

class CropBlackBordersRGB(object):
    '''
    功能说明:
        对超声图像周边的无用黑色边框区域进行裁剪
    '''
    def __init__(self, threshold=10, black_ratio_threshold=0.5, min_width=200, min_height=200):
        self.threshold = threshold #判定黑色像素阈值
        self.black_ratio_threshold = black_ratio_threshold #判定黑色区域阈值
        self.min_width = min_width  # 最小宽度阈值
        self.min_height = min_height  # 最小高度阈值

    def __call__(self, img):
        img_np = np.array(img)
        height, width, _ = img_np.shape

        # 裁剪左右黑边
        left_crop = 0
        for x in range(width // 2):  # 从左到中间扫描
            column = img_np[:, x]
            black_pixels = np.sum((column[:, 0] <= self.threshold) & (column[:, 1] <= self.threshold) & (column[:, 2] <= self.threshold))
            if black_pixels / height > self.black_ratio_threshold:
                left_crop = x + 1
            else:
                break

        right_crop = width
        for x in range(width - 1, width // 2, -1):  # 从右到中间扫描
            column = img_np[:, x]
            black_pixels = np.sum((column[:, 0] <= self.threshold) & (column[:, 1] <= self.threshold) & (column[:, 2] <= self.threshold))
            if black_pixels / height > self.black_ratio_threshold:
                right_crop = x - 1
            else:
                break

        # 裁剪上下黑边
        top_crop = 0
        for y in range(height // 2):  # 从上到中间扫描
            row = img_np[y, :]
            black_pixels = np.sum((row[:, 0] <= self.threshold) & (row[:, 1] <= self.threshold) & (row[:, 2] <= self.threshold))
            if black_pixels / width > self.black_ratio_threshold:
                top_crop = y + 1
            else:
                break

        bottom_crop = height
        for y in range(height - 1, height // 2, -1):  # 从下到中间扫描
            row = img_np[y, :]
            black_pixels = np.sum((row[:, 0] <= self.threshold) & (row[:, 1] <= self.threshold) & (row[:, 2] <= self.threshold))
            if black_pixels / width > self.black_ratio_threshold:
                bottom_crop = y - 1
            else:
                break

        cropped_img_np = img_np[top_crop:bottom_crop, left_crop:right_crop]

        # 如果裁剪后的图像尺寸小于最小阈值，则返回原图
        if cropped_img_np.shape[1] < self.min_width or cropped_img_np.shape[0] < self.min_height:
            return img

        cropped_img = transforms.ToPILImage()(cropped_img_np)
        return cropped_img

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform, cutmix=False, cutmix_prob=0.5, CutBlack=False):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_cutmix = cutmix
        self.cutmix_prob = cutmix_prob
        self.apply_cutblack = CutBlack
        self.cut_black_fn = CropBlackBordersRGB()
        self.cutmix_fn = CutMix(prob=cutmix_prob)
        self.image_paths = []
        self.labels = []
        self.bboxes = []

        images_folder_path = os.path.join(root_dir, 'cla')
        labels_folder_path = os.path.join(root_dir, 'labels')

        for image_name in os.listdir(images_folder_path):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(images_folder_path, image_name)
                label_file_path = os.path.join(labels_folder_path, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

                with open(label_file_path, 'r') as label_file:
                    data = list(map(float, label_file.readline().strip().split()))
                    label = int(data[0]) - 1  # 标签
                    bbox = data[1:]  # 边框信息

                self.image_paths.append(image_path)
                self.labels.append(label)
                self.bboxes.append(bbox)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和标签
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        image = Image.open(image_path).convert('RGB')

        if self.apply_cutmix:
            other_idx = random.randint(0, len(self.image_paths) - 1)
            other_image, other_label, other_bbox = self.load_image_and_label(other_idx)
            image = self.cutmix_fn(image, other_image, bbox, other_bbox)

        if self.apply_cutblack:
            image = self.cut_black_fn(image)

        image = self.transform(image)

        return image, label

    def load_image_and_label(self, idx):
        """加载另一张图像及其标签和边框信息。"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label, bbox


random_erasing = transforms.RandomErasing(
    p=0.9,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value=0,
    inplace=False
)

test_transforms = transforms.Compose([
    transforms.CenterCrop(672),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms = transforms.Compose([
    transforms.CenterCrop(672),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 包含旋转、平移、缩放和剪切
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.9, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])


