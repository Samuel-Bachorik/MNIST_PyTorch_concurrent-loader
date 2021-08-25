import torch
from torch import nn, optim
from mnist_model import Model
from dataset_loader import ImagesLoader

paths = []
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/0/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/1/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/2/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/3/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/4/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/5/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/6/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/7/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/8/")
paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/9/")



if __name__ == '__main__':

    loader = ImagesLoader(128,paths)
    dataset = loader.get_dataset()

    model = Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    lossum = 0
    epoch = 10
    print("Training started...")
    for i in range(epoch):

        for images, labels in (zip(*dataset)):

            images = images.to(model.device)
            labels = labels.to(model.device)

            iamges, labels = images.cuda(), labels.cuda()
            y = model(images)

            loss = criterion(y, labels)
            lossnumber = float(loss.data.cpu().numpy())
            lossum += lossnumber

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Average loss for epoch ",i+1, " - ",lossum / len(dataset[0]))
        lossum = 0


    PATH = './MNIST-MY.pth'
    torch.save(model.state_dict(), PATH)
    print("Model saved")
