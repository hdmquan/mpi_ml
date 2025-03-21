import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.data.module import MPIDataModule

datamodule = MPIDataModule()
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# Initialize scaler for feature standardization
scaler = StandardScaler()

# Initialize SGDRegressor with good default settings
base_sgd = SGDRegressor(
    loss="huber",  # Huber loss is robust to outliers
    learning_rate="adaptive",  # Automatically decreases learning rate
    eta0=0.01,  # Initial learning rate
    penalty="l2",  # L2 regularization to prevent overfitting
    alpha=0.0001,  # Regularization strength
    max_iter=1000,
    tol=1e-3,
    warm_start=True,  # Enable warm start for partial_fit
    random_state=37,
)

# Create separate models for each output
models = [SGDRegressor(**base_sgd.get_params()) for _ in range(6)]


def extract_features_labels(batch):
    inputs, (mmr, _, _), _ = batch
    # logger.debug(f"inputs: {inputs.shape} | mmr: {mmr.shape}")
    # inputs: torch.Size([1, 8, 48, 384, 576]) | mmr: torch.Size([1, 6, 48, 384, 576])
    inputs = inputs.numpy()
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    inputs = inputs.transpose(0, 2, 1)
    inputs = inputs.reshape(-1, inputs.shape[2])

    mmr = mmr.numpy()
    mmr = mmr.reshape(mmr.shape[0], mmr.shape[1], -1)
    mmr = mmr.transpose(0, 2, 1)
    mmr = mmr.reshape(-1, mmr.shape[2])

    return inputs, mmr


# First batch used to initialize the scaler
first_batch = next(iter(train_loader))
X_init, _ = extract_features_labels(first_batch)
scaler.fit(X_init)

# Training loop
for epoch in range(5):  # Multiple epochs over the data
    for batch in train_loader:
        X_train, y_train = extract_features_labels(batch)

        # Standardize features
        X_train = scaler.transform(X_train)

        logger.debug(f"X: {X_train.shape} | y: {y_train.shape}")

        # Partial fit each model
        for i, model in enumerate(models):
            model.partial_fit(X_train, y_train[:, i])


def evaluate(loader, name):
    y_true, y_pred = [], []
    for batch in loader:
        X, y = extract_features_labels(batch)
        X = scaler.transform(X)  # Standardize features

        # Predict using all models
        batch_preds = np.column_stack([model.predict(X) for model in models])
        y_true.append(y)
        y_pred.append(batch_preds)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mse_per_output = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    avg_mse = mean_squared_error(y_true, y_pred)
    print(f"{name} Average MSE: {avg_mse:.4f}")
    for i, mse in enumerate(mse_per_output):
        print(f"{name} MSE (output {i}): {mse:.4f}")


evaluate(val_loader, "Validation")
evaluate(test_loader, "Test")
