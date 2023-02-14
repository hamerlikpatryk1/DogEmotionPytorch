import glob
import cv2
import matplotlib.image as mpimg
import numpy as np
import torch
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from random import sample


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "images/"
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        # print(self.data)
        self.class_map = {"angry": 0, "happy": 1, "relaxed": 2, "sad": 3}
        self.img_dim = (384, 384)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id.float()


# def append_my_list(self, drawn_list, my_list):
#     drawn_list.append(np.random.permutation(my_list)[:10])
#     return drawn_list

def main():
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("done")
    label_counters = {"angry": 0, "happy": 0, "relaxed": 0, "sad": 0}
    for label in dataset.class_map.keys():
        for record in dataset.data:
            if record[1] == label:
                label_counters[label] += 1
    print(label_counters)

    class_counts = dataset.class_map.copy()
    for label in class_counts.keys():
        class_counts[label] = sum([1 if record[1] == label else 0 for record in dataset.data])
    print(class_counts)

    angry_list = [photo for photo in dataset.data if photo[1] == "angry"]
    happy_list = [photo for photo in dataset.data if photo[1] == "happy"]
    relaxed_list = [photo for photo in dataset.data if photo[1] == "relaxed"]
    sad_list = [photo for photo in dataset.data if photo[1] == "sad"]

    drawn_photos = []
    drawn_photos.extend(sample(angry_list, 10))
    drawn_photos.extend(sample(happy_list, 10))
    drawn_photos.extend(sample(relaxed_list, 10))
    drawn_photos.extend(sample(sad_list, 10))

    print(drawn_photos)
    print(drawn_photos[0][0])

    # for photo in dataset.data[]:
    # img = mpimg.imread(drawn_photos[0][0])
    i = 0
    rows = len(label_counters)
    columns = (int(len(drawn_photos) / rows))
    fig, ax = plt.subplots(nrows=rows, ncols=columns)
    for row in range(rows):
        for col in range(columns):
            ax[row, col].imshow(mpimg.imread(drawn_photos[i][0]))
            ax[row, col].set_xlabel(drawn_photos[i][1])
            i += 1

    # ax[0, 0].imshow(drawn_photos[0])
    # for ax.set_title(str())

    plt.subplot_tool()
    plt.show()


if __name__ == "__main__":
    main()
