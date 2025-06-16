import torchvision.transforms as transforms

def transform():
    return transforms.Compose([
        transforms.CenterCrop(207),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])