import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps

# 对超声图像周边的无用黑色边框区域进行裁剪
class CropBlackBordersRGB(object):
    def __init__(self, threshold=10, black_ratio_threshold=0.5, min_width=200, min_height=200):
        self.threshold = threshold
        self.black_ratio_threshold = black_ratio_threshold
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

        if cropped_img_np.shape[1] < self.min_width or cropped_img_np.shape[0] < self.min_height:
            return img

        return transforms.ToPILImage()(cropped_img_np)


class CutMix:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img1, img2):
        if random.random() < self.prob:
            return self.apply_cutmix_aug(img1, img2)
        else:
            return img1

    def apply_cutmix_aug(self, img1, img2):
        w, h = img1.size
        x = random.randint(0, w)
        y = random.randint(0, h)
        bw = random.randint(w // 2, w)
        bh = random.randint(h // 2, h)

        img2 = ImageOps.fit(img2, img1.size, method=Image.BILINEAR)

        mask = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, min(x + bw, w), min(y + bh, h)], fill=0)

        return Image.composite(img1, img2, mask)

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None, apply_cutmix=False, cutmix_prob=0.5, apply_cutblack=False):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_cutmix = apply_cutmix
        self.cutmix_prob = cutmix_prob
        self.apply_cutblack = apply_cutblack
        self.cut_black_fn = CropBlackBordersRGB()
        self.cutmix_fn = CutMix(prob=cutmix_prob)
        self.image_paths = []
        self.labels = []

        images_folder_path = os.path.join(root_dir, 'cla')
        labels_folder_path = os.path.join(root_dir, 'labels')
        feature_folders = [folder for folder in os.listdir(labels_folder_path)
                           if os.path.isdir(os.path.join(labels_folder_path, folder))]

        for image_name in os.listdir(images_folder_path):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(images_folder_path, image_name)

                labels = []
                all_labels_found = True

                for feature_folder in feature_folders:
                    feature_folder_path = os.path.join(labels_folder_path, feature_folder)
                    label_file_path = os.path.join(feature_folder_path,
                                                   image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

                    try:
                        with open(label_file_path, 'r') as label_file:
                            first_line = label_file.readline().strip()
                            if first_line:
                                label = int(first_line.split()[0])
                                labels.append(label)
                            else:
                                all_labels_found = False
                                break
                    except FileNotFoundError:
                        all_labels_found = False
                        break

                if all_labels_found:
                    self.image_paths.append(image_path)
                    self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        labels = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.apply_cutmix and random.random() < self.cutmix_prob:
            other_idx = random.randint(0, len(self.image_paths) - 1)
            other_image = self.load_image(other_idx)
            image = self.cutmix_fn(image, other_image)

        if self.apply_cutblack:
            image = self.cut_black_fn(image)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def load_image(self, idx):
        """用于cutmix实现，该函数功能为加载另一张图像。"""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return image


random_erasing = transforms.RandomErasing(
    p=0.9,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value=0,
    inplace=False
)

train_transforms = transforms.Compose([
    transforms.CenterCrop(672),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], random_erasing),
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(672),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms_new = transforms.Compose([
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

