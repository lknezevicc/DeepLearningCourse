import torch
import matplotlib.pyplot as plt

# Initialization
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(42)
x = torch.linspace(-2, 2, 3000, device=device, dtype=dtype)
y = torch.sin(x**2) + 2 * x**3 + 3

plt.plot(x.cpu(), y.cpu())
plt.title("y = sin(x^2) + 2 * x^3 + 3")
plt.grid()
plt.show()

p = torch.tensor([1, 2, 3, 4], device=device)
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
  torch.nn.Linear(4, 1),
  torch.nn.Flatten(0, 1)
).to(device=device)

loss_fn = torch.nn.MSELoss(reduction='sum').to(device=device)

learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(5000):
  y_pred = model(xx)
  loss = loss_fn(y_pred, y)
  if t % 1000 == 0:
    print(f"Epoch {t}, Loss: {loss.item()}")

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if t % 1000 == 0:
    linear_layer = model[0]
    a = linear_layer.bias.item()
    b = linear_layer.weight[0, 0].item()
    c = linear_layer.weight[0, 1].item()
    d = linear_layer.weight[0, 2].item()
    
    y_graph = a * x**3 + b * x**2 + c * x + d
    plt.plot(x.cpu(), y.cpu(), label='Original')
    plt.plot(x.cpu(), y_graph.cpu().detach(), label='Predikcija')
    plt.title(f"Epoch {t}")
    plt.legend()
    plt.grid()
    plt.show()
