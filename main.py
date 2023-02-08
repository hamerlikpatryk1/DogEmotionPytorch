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
    # for imgs, labels in data_loader:
    #   print("Batch of images has shape: ", imgs.shape)
    #   print("Batch of labels has shape: ", labels.shape)
    # return
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

    angry_list = []
    happy_list = []
    relaxed_list = []
    sad_list = []

    for photo in dataset.data:
        if photo[1] == "angry":
            angry_list.append(photo)
        if photo[1] == "happy":
            happy_list.append(photo)
        if photo[1] == "relaxed":
            relaxed_list.append(photo)
        if photo[1] == "sad":
            sad_list.append(photo)
    # print(angry_list)

    # print(dataset.data) structure [[][][][][]...]
    drawn_photos = []
    drawn_photos.extend(sample(angry_list, 10))
    drawn_photos.extend(sample(happy_list, 10))
    drawn_photos.extend(sample(relaxed_list, 10))
    drawn_photos.extend(sample(sad_list, 10))

    print(drawn_photos)
    print(drawn_photos[0][0])
    img = mpimg.imread(drawn_photos[0][0])

    fig, ax = plt.subplots(nrows=4, ncols=10)
    ax[0, 0].imshow(img)
    # ax[0, 0].imshow(drawn_photos[0])
    # for ax.set_title(str())

    plt.tight_layout()
    plt.show()

    # print(drawn_photos)  # can also use choice function
    # links_list = []
    ##label_list = []
    # for photo in dataset.data[]:

    #  links_list.append(photo[0])  # links
    # label_list.append(photo[1])  # labels
    # print("links= ", links_list, " labels= ", label_list)  # lists are ordered


if __name__ == "__main__":
    main()
