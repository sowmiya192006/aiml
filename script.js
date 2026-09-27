import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "smart_farming_data.csv"     # path to the Kaggle Smart Farming IoT dataset
TARGET_COLUMN = "yield"                   # crop yield column (rename to match actual dataset)
NUM_EDGE_SERVERS = 4
RANDOM_STATE = 42
MODEL_OUTPUT_DIR = Path("edge_models")
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the smart farming dataset. Falls back to a synthetic dataset
    (for demo/testing purposes) if the real file is not found."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded real dataset from {path}, shape={df.shape}")
    except FileNotFoundError:
        print(f"'{path}' not found - generating a synthetic demo dataset instead.")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 2000
        df = pd.DataFrame({
            "soil_moisture": rng.uniform(10, 60, n),
            "temperature": rng.uniform(15, 40, n),
            "humidity": rng.uniform(20, 90, n),
            "rainfall": rng.uniform(0, 300, n),
            "n_content": rng.uniform(0, 140, n),
            "p_content": rng.uniform(0, 140, n),
            "k_content": rng.uniform(0, 200, n),
            "ph": rng.uniform(4.5, 8.5, n),
        })
        # synthetic yield as a noisy function of the features
        df[TARGET_COLUMN] = (
            0.3 * df["soil_moisture"]
            + 0.5 * df["rainfall"] / 10
            + 0.2 * df["n_content"]
            + 0.1 * df["k_content"]
            - 0.4 * abs(df["ph"] - 6.5) * 10
            + rng.normal(0, 5, n)
        )
    return df


def preprocess(df: pd.DataFrame):
    """Basic cleaning: drop missing values, separate features/target."""
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_across_edge_servers(X: pd.DataFrame, y: pd.Series, num_servers: int, seed: int):
    """Shuffle then split the training data into `num_servers` roughly equal,
    non-overlapping shards - one per simulated edge server."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_shuffled = X.iloc[perm].reset_index(drop=True)
    y_shuffled = y.iloc[perm].reset_index(drop=True)

    shards = np.array_split(np.arange(len(X_shuffled)), num_servers)
    server_data = []
    for shard_idx in shards:
        X_shard = X_shuffled.iloc[shard_idx].reset_index(drop=True)
        y_shard = y_shuffled.iloc[shard_idx].reset_index(drop=True)
        server_data.append((X_shard, y_shard))
    return server_data


def local_training_loop(server_data, random_state=RANDOM_STATE):
    """Train one Random Forest Regressor per edge server on its own local shard.

    Returns a list of dicts with the trained model, MAE, and shard sizes.
    """
    results = []

    for server_id, (X_local, y_local) in enumerate(server_data, start=1):
        # Local train/validation split - stays entirely on this "edge server"
        X_train, X_val, y_train, y_val = train_test_split(
            X_local, y_local, test_size=0.2, random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)

        model_path = MODEL_OUTPUT_DIR / f"edge_server_{server_id}_rf_model.joblib"
        joblib.dump(model, model_path)

        print(f"[Edge Server {server_id}] "
              f"train_size={len(X_train)}, val_size={len(X_val)}, MAE={mae:.4f} "
              f"-> saved to {model_path}")

        results.append({
            "server_id": server_id,
            "model": model,
            "mae": mae,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "model_path": str(model_path),
        })

    return results


def summarize(results):
    maes = [r["mae"] for r in results]
    print("\n--- Local Training Summary ---")
    for r in results:
        print(f"Edge Server {r['server_id']}: MAE = {r['mae']:.4f} "
              f"(n_train={r['train_size']}, n_val={r['val_size']})")
    print(f"Average MAE across all edge servers: {np.mean(maes):.4f}")
    print(f"All {len(results)} local models saved in '{MODEL_OUTPUT_DIR}/' "
          f"ready for global aggregation.")


def main():
    df = load_dataset(DATA_PATH)
    X, y = preprocess(df)
    server_data = split_across_edge_servers(X, y, NUM_EDGE_SERVERS, RANDOM_STATE)
    results = local_training_loop(server_data, RANDOM_STATE)
    summarize(results)
    return results


if __name__ == "__main__":
    main()
