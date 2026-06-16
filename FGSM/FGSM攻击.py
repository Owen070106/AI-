import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def train():
    mnist_datasets=datasets.MNIST(
        './data/mnist',train=True,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )
    train_loader=torch.utils.data.DataLoader(mnist_datasets,batch_size=10,shuffle=True)
    model=Net()
    model = model.to(device)
    critertion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)

    epoch=5
    for epoch_idx in range(epoch):
        model.train()
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pre = model(x)
            loss=critertion(y_pre,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_correct += (torch.argmax(y_pre, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        print(f"epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.2f}, time:{time.time() - start:.2f}s")
    torch.save(model.state_dict(), "./data/lenet_mnist_model.pth")


def eval():
    mnist_datasets = datasets.MNIST(
        './data/mnist', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )
    dataloder = torch.utils.data.DataLoader(mnist_datasets, batch_size=10, shuffle=False)
    model = Net()
    model = model.to(device)
    model.load_state_dict(torch.load("./data/lenet_mnist_model.pth"))
    total_correct, total_samples = 0, 0
    for x, y in dataloder:
        x, y = x.to(device), y.to(device)
        model.eval()
        y_pre = model(x)
        total_correct += (torch.argmax(y_pre, dim=-1) == y).sum()
        total_samples += len(y)
    print(f'Acc: {total_correct / total_samples:.2f}')


def fgsm_attack(image,epsilon,data_grad):
    sign_data_grad=data_grad.sign()
    perturbed_image=image+epsilon*sign_data_grad
    perturbed_image=torch.clamp(perturbed_image,0,1)
    return perturbed_image

def denorm(batch, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test(model,device,test_loader,epsilon):
    correct=0
    adv_examples=[]
    model.eval()
    for data,target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad=True
        output=model(data)
        init_pred=output.max(1,keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        if data.grad is not None:
            data.grad.zero_()
        loss.backward()
        data_grad=data.grad.data
        data_denorm=denorm(data)
        perturbed_data=fgsm_attack(data_denorm,epsilon,data_grad)
        preturbed_data_normalized=transforms.Normalize((0.1307,),(0.3081,))(perturbed_data)
        output=model(preturbed_data_normalized)
        final_pred=output.max(1,keepdim=True)[1]
        if final_pred.item()==target.item():
            correct += 1
            if epsilon==0 and len(adv_examples)<5:
                adv_ex=perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(),final_pred.item(),adv_ex))
        else:
            if len(adv_examples)<5:
                adv_ex=perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct / float(len(test_loader))
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples


if __name__ == '__main__':
    accuracies=[]
    examples=[]
    epsilons=[0,0.05,0.1,0.15,0.2,0.25,0.3]
    model=Net().to(device)
    model.load_state_dict(torch.load("./data/lenet_mnist_model.pth"))
    text_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])),
        batch_size=1, shuffle=True)
    for eps in epsilons:
        acc,ex=test(model,device,text_loader,eps)
        accuracies.append(acc)
        examples.append(ex)
    # plt.figure(figsize=(5, 5))
    # plt.plot(epsilons, accuracies, "*-")
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.xticks(np.arange(0, .35, step=0.05))
    # plt.title("Accuracy vs Epsilon")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.show()
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()



