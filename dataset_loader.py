import os, os.path
from PIL import Image
import numpy
import random
import torch
import concurrent.futures
import math
class ImagesLoader:
    def __init__(self, batch,paths):
        self.batch = batch
        self.paths = paths

    def loop(self):
        labels0 = numpy.zeros(self.batch, dtype=numpy.int64)
        imgs = numpy.zeros((self.batch, 1, 28, 28), dtype=numpy.float32)

        for i in range(self.batch):

            rand_path = random.randint(0, 9)
            path = self.paths[rand_path]
            labels0[i] = rand_path

            f = os.listdir(path)
            x = Image.open(os.path.join(path, f[random.randint(0, len(f) - 1)]))

            img = numpy.asarray(x)

            img = numpy.expand_dims(img, axis=0)
            g = (img / 255.0).astype(numpy.float32)

            g = g.squeeze(0)

            imgs[i] = g.copy()

        return labels0, imgs

    def get_dataset(self):
        print("Loading dataset...")
        epoch = 60000/self.batch
        imgs2 = numpy.zeros((math.ceil(epoch), self.batch, 1, 28, 28), dtype=numpy.float32)
        labels = numpy.zeros((math.ceil(epoch), self.batch), dtype=numpy.int64)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [None] * math.ceil(epoch)
            for x in range(math.ceil(epoch)):
                results[x] = executor.submit(self.loop)


            counter = 0
            for f in concurrent.futures.as_completed(results):
                imgs2[counter], labels[counter] = f.result()[1], f.result()[0]
                counter += 1


        return torch.from_numpy(imgs2), torch.from_numpy(labels)




















"""paths = []
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/0/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/1/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/2/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/3/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/4/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/5/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/6/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/7/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/8/")
paths.append("C:/Users/Samuel/PycharmProjects/Condapytorch/MNIST/MNIST - JPG - training/9/")

rows, cols = (100, 2)
dataset = [[0]*cols]*rows

labels = numpy.zeros(64, dtype=int)

counter = 0
counter2 = 0

imgs = numpy.zeros((64, 1, 28,28),dtype=numpy.float32)
#99 krat


for e in range(5):

    for i in range(64):

        path = random.choice(paths)
        index = paths.index(path)

        labels[i] = index

        f = os.listdir(path)
        x = Image.open(os.path.join(path,f[counter2]))

        img = numpy.array(x)
        img = numpy.expand_dims(img, axis=0)
        g = (img / 255.0).astype(numpy.float32)


        imgs[i] = g.copy()

        counter+=1

        if counter >= 10:
            counter = 0
            counter2 +=1


    dataset[e][0] = torch.from_numpy(imgs)
    dataset[e][1] = torch.from_numpy(labels)

print(dataset[1][0].shape)


"""



