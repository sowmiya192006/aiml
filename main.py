import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# Simulate 4 Edge Servers
edge_servers = 4
edge_data = []

size = len(X_train) // edge_servers

for i in range(edge_servers):
    start = i * size
    end = (i + 1) * size if i < edge_servers - 1 else len(X_train)

    edge_data.append((
        X_train.iloc[start:end],
        y_train.iloc[start:end]
    ))

# Local training on each Edge Server
models = []

for i, (Xe, ye) in enumerate(edge_data):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(Xe, ye)
    models.append(model)

    print(f"Edge Server {i + 1}: Training Completed")

# Test Client 1 model
pred = models[0].predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("MAE:", mae)
