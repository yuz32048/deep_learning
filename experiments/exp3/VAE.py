from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 1.数据导入与处理
train_datasets = datasets.MNIST(
    root='../exp2/datasets/train_datasets',
    train=True,
    transform=transforms.ToTensor(),
)

test_datasets = datasets.MNIST(
    root='../exp2/datasets/test_datasets',
    train=False,
    transform=transforms.ToTensor(),
)

train_loader = DataLoader(dataset=train_datasets, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_datasets, batch_size=64, shuffle=True)

# 2.构建VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器结构
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
        )

        # 生成均值
        self.mu = nn.Linear(400, 20)
        # 生成对数方差（网络输出是任意实数，而方差是正数，因此采用对数形式进行转换）
        self.log_var = nn.Linear(400, 20)

        # 解码器结构
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    # 定义编码网络
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    # 定义重参数化
    def reparameterize(self, mu, log_var):
        # 由对数方差得到标准差
        std = torch.exp(0.5*log_var)
        # 创建与std相同形状的服从正态分布的随机张量
        eps = torch.randn_like(std)
        return mu + eps*std

    # 定义解码网络
    def decode(self, z):
        return self.decoder(z)

    # 定义正向传播过程
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output, mu, log_var

# 3.定义损失函数
def vae_loss(x, output, mu, log_var, beta=1.0, method="mse"):
    '''

    Args:
        x: 模型输入
        output: 模型输出
        mu: 编码器输出均值
        log_var: 编码器输出对数方差
        beta: 调整重构损失和 KL 散度权重

    Returns:
        每张图片的损失，重构损失，KL散度
    '''
    # print(x.shape)
    # print(output.shape)
    x = x.reshape(x.shape[0], -1)
    # 重建损失默认采用 MSE
    criterion = nn.MSELoss(reduction="sum")

    # 重构损失采用BCE
    if method == "bce":
        criterion = nn.BCELoss(reduction="sum")

    recon_loss = criterion(output, x)
    # KL散度
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    batch_size = x.shape[0]
    loss = (recon_loss + beta*kl_loss)/batch_size

    return loss, recon_loss.item()/batch_size, kl_loss.item()/batch_size

# 4.模型训练
VAEModel = VAE()
optimizer = optim.Adam(VAEModel.parameters(), lr=0.001)

recon_loss_list = []
kl_loss_list = []
loss_list = []
epoch_nums = 20
for epoch in range(epoch_nums):
    VAEModel.train()
    samples_each = 0
    loss_each = 0
    kl_loss_each = 0
    recon_loss_each = 0
    for x_train, y_train in train_loader:
        y_pred, mu, log_var = VAEModel(x_train)
        loss, recon_loss, kl_loss = vae_loss(x_train, y_pred, mu, log_var, method = "bce")

        samples_each += len(y_train)
        loss_each += loss.item()*len(y_train)
        kl_loss_each += kl_loss*len(y_train)
        recon_loss_each += recon_loss*len(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_list.append(loss_each/samples_each)
    recon_loss_list.append(recon_loss_each/samples_each)
    kl_loss_list.append(kl_loss_each/samples_each)
    print(f"第{epoch+1}/{epoch_nums}轮训练，Loss：{loss_list[-1]:.4f}，recon_loss:{recon_loss_list[-1]:.4f}，kl_loss:{kl_loss_list[-1]:.4f}")

# 5.模型测试
# 模型生成能力测试
VAEModel.eval()
with torch.no_grad():
    # 从潜在空间中随机采样10个变量
    z = torch.randn(10,20)
    output = VAEModel.decode(z)
# print(output.shape())
# 重新整理输出形状，使得输出图片
output = output.reshape(output.shape[0], 1, 28, 28)
# print(output.shape())
# 保存图片
save_image(output, "./VAE_generate_bce.png", nrow=5)

# 模型重建能力测试
with torch.no_grad():
    samples_test = 0
    loss_test = 0
    kl_loss_test = 0
    recon_loss_test = 0
    for x_test, y_test in test_loader:
        y_pred, mu, log_var = VAEModel(x_test)
        loss, recon_loss, kl_loss = vae_loss(x_test, y_pred, mu, log_var, method="bce")

        samples_test += len(y_test)
        loss_test += loss.item()*len(y_test)
        recon_loss_test += recon_loss*len(y_test)
        kl_loss_test += kl_loss*len(y_test)

    vae_loss_test = loss_test/samples_test
    vae_recon_loss_test = recon_loss_test/samples_test
    vae_kl_loss_test = kl_loss_test/samples_test
    print(f"模型重建能力测试，Loss：{vae_loss_test:.4f}，recon_loss:{vae_recon_loss_test:.4f}，kl_loss:{vae_kl_loss_test:.4f}")
    print(f"针对每个像素的损失，Loss：{vae_loss_test/784:.4f}，recon_loss:{vae_recon_loss_test/784:.4f}，kl_loss:{vae_kl_loss_test/784:.4f}")

    # 可视化测试
    x_test, _ = next(iter(test_loader))
    x_test = x_test[:5]
    y_pred, mu, log_var = VAEModel(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 28, 28)
    pic = torch.cat((x_test, y_pred), dim=0)
    save_image(pic, "./VAE_visualize_bce.png", nrow=5)

# 6.绘图
# 损失变化图
# (1) 重构损失变化图
plt.figure()
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.title("重构损失变化图")
plt.xlabel("迭代轮次epoch")
plt.ylabel("Loss")
plt.plot(recon_loss_list, label="recon_loss", color = "black")
plt.legend()
plt.savefig("./VAE_recon_loss_bce.png")
plt.show()

# (2) kl散度变化图
plt.figure()
plt.title("KL散度变化图")
plt.xlabel("迭代轮次epoch")
plt.ylabel("KL散度")
plt.plot(kl_loss_list, label="kl_loss", color = "red")
plt.legend()
plt.savefig("./VAE_kl_loss_bce.png")
plt.show()
