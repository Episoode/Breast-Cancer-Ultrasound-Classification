import os
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps

test_transforms = transforms.Compose([
    transforms.CenterCrop(672),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 对超声图像周边的无用黑色边框区域进行裁剪
class CropBlackBordersRGB(object):
    def __init__(self, threshold=10, black_ratio_threshold=0.5, min_width=200, min_height=200):
        self.threshold = threshold # 扫描宽度阈值
        self.black_ratio_threshold = black_ratio_threshold # 判定黑色阈值
        self.min_width = min_width
        self.min_height = min_height

    def __call__(self, img):
        img_np = np.array(img)
        height, width, _ = img_np.shape

        # 裁剪左右黑边
        left_crop, right_crop = 0, width
        for x in range(width // 2):
            column = img_np[:, x]
            black_pixels = np.sum((column[:, 0] <= self.threshold) & (column[:, 1] <= self.threshold) & (column[:, 2] <= self.threshold))
            if black_pixels / height > self.black_ratio_threshold:
                left_crop = x + 1
            else:
                break

        for x in range(width - 1, width // 2, -1):
            column = img_np[:, x]
            black_pixels = np.sum((column[:, 0] <= self.threshold) & (column[:, 1] <= self.threshold) & (column[:, 2] <= self.threshold))
            if black_pixels / height > self.black_ratio_threshold:
                right_crop = x - 1
            else:
                break

        # 裁剪上下黑边
        top_crop, bottom_crop = 0, height
        for y in range(height // 2):
            row = img_np[y, :]
            black_pixels = np.sum((row[:, 0] <= self.threshold) & (row[:, 1] <= self.threshold) & (row[:, 2] <= self.threshold))
            if black_pixels / width > self.black_ratio_threshold:
                top_crop = y + 1
            else:
                break

        for y in range(height - 1, height // 2, -1):
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

        return transforms.ToPILImage()(cropped_img_np)


class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=test_transforms, apply_cutblack=False):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_cutblack = apply_cutblack # 是否进行黑色边框裁剪
        self.cut_black_fn = CropBlackBordersRGB()
        self.image_paths = []

        for image_name in os.listdir(root_dir):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(root_dir, image_name)
                self.image_paths.append(image_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)  # 获取图像文件名
        image = Image.open(image_path).convert('RGB')

        if self.apply_cutblack:
            image = self.cut_black_fn(image)

        if self.transform:
            image = self.transform(image)

        return image_name, image