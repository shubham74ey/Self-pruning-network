import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += gates.sum()
    return loss

transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

def train_model(lambda_sparse):
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):  
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_cls = criterion(outputs, labels)
            loss_sp = sparsity_loss(model)
            loss = loss_cls + lambda_sparse * loss_sp
            loss.backward()
            optimizer.step()

    return model

def test_accuracy(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total


def sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total

lambdas = [1e-5, 1e-4, 1e-3,]

results = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model = train_model(lam)
    acc = test_accuracy(model)
    sp = sparsity(model)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sp:.2f}%")
    results.append((lam, acc, sp))

all_gates = []
for m in model.modules():
    if isinstance(m, PrunableLinear):
        g = torch.sigmoid(m.gate_scores).detach().numpy()
        all_gates.extend(g.flatten())
plt.hist(all_gates, bins=50)
plt.title("Gate Distribution")
plt.show()
print("\nFinal Results:")
for r in results:
    print(r)
