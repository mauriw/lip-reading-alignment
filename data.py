import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import matplotlib as plt
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

class LipReadingData(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        return image, label

if __name__ == '__main__':
    print("hello!")
    training_data = LipReadingData('data/labels.csv', 'data/images')
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)    
    img, label = next(iter(train_dataloader))
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.show()
    print(f"Label: {label}")


