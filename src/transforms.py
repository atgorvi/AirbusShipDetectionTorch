import albumentations as A

import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2


def get_train_aug(img_size: int):
    crop_size = int(img_size * 0.75)
    train_transform = [

        A.Perspective(),
        A.Flip(0.5),
        A.OneOf([A.Blur(blur_limit=3, p=1),
                 A.MotionBlur(blur_limit=3, p=1),
                 ], p=0.5),
        A.OneOf([
            A.CropNonEmptyMaskIfExists(crop_size, crop_size),
            A.RandomCrop(crop_size, crop_size),
            A.CenterCrop(crop_size, crop_size)], p=0.5),
        # A.Normalize(),
        # A.ElasticTransform(p=1),
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_val_aug(img_size: int):
    val_transform = [
        A.Resize(img_size, img_size),
        # A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(val_transform)

def get_test_aug(img_size: int):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),

    ])
    return test_transform


if __name__ == "__main__":
    pass