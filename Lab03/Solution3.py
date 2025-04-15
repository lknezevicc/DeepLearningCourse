import torch
import matplotlib.pyplot as plt
import pandas as pd

csv_train_path = 'random-linear-regression/train.csv'
csv_test_path = 'random-linear-regression/test.csv'

df_train = pd.read_csv(csv_train_path)
df_test = pd.read_csv(csv_test_path)

df_train.dropna(inplace=True)

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_train = torch.tensor(df_train['x'].values, dtype=dtype, device=device).unsqueeze(1)
x_test = torch.tensor(df_test['x'].values, dtype=dtype, device=device).unsqueeze(1)

y_train = torch.tensor(df_train['y'].values, dtype=dtype, device=device).unsqueeze(1)
y_test = torch.tensor(df_test['y'].values, dtype=dtype, device=device).unsqueeze(1)

model = torch.nn.Sequential(
  torch.nn.Linear(1, 1)
).to(device=device)

loss_fn = torch.nn.MSELoss(reduction='mean').to(device=device)

learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(5000):
  y_pred = model(x_train)
  loss = loss_fn(y_pred, y_train)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if t % 1000 == 0:
    linear_layer = model[0]
    a = linear_layer.bias.item()
    b = linear_layer.weight[0, 0].item()
    print(f"Epoch {t}, Loss: {loss.item()}")

    y_graph = b * x_test + a

    plt.scatter(x_test.cpu(), y_test.cpu(), label='Original')
    plt.plot(x_test.cpu(), y_graph.cpu().detach(), label='Predikcija', color='red')
    plt.title(f"Epoch {t}")
    plt.legend()
    plt.grid()
    plt.show()
