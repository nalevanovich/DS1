"""
models/lstm_model.py
─────────────────────
LSTM (PyTorch) для агрегированных продаж Walmart M5.
Использует скользящее окно SEQ_LEN дней → предсказываем следующий.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


class _LSTMNet(nn.Module):
    def __init__(self, hidden1=64, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden1, batch_first=True, num_layers=1)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden1, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out)))


def _make_sequences(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_predict(sales_all: np.ndarray,
                test_size: int,
                seq_len: int = 28,
                epochs: int = 60,
                patience: int = 10,
                batch_size: int = 32,
                lr: float = 1e-3,
                plot_loss: bool = True,
                save: bool = False):
    """
    Обучает LSTM и возвращает прогноз.

    Parameters
    ----------
    sales_all : полный ряд суммарных продаж (train + test)
    test_size : размер тестовой выборки
    seq_len   : длина входного окна (28 = 4 недели)

    Returns
    -------
    (pred, y_test_lstm) — оба в оригинальных единицах (шт.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Устройство: {device}")

    # Заменяем нули (дни закрытия магазина) на медиану
    sales_clean = sales_all.copy().astype(np.float32)
    median_val  = np.median(sales_clean[sales_clean > 0])
    sales_clean[sales_clean == 0] = median_val
    print(f"  Заменено нулей: {(sales_all == 0).sum()} → медиана {median_val:.0f}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(sales_clean.reshape(-1, 1)).flatten()

    X_seq, y_seq = _make_sequences(scaled, seq_len)
    split = len(X_seq) - test_size

    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, _    = y_seq[:split], y_seq[split:]

    val_n = max(test_size, int(len(X_tr) * 0.1))  # минимум test_size дней
    X_val, y_val = X_tr[-val_n:], y_tr[-val_n:]
    X_tr, y_tr   = X_tr[:-val_n], y_tr[:-val_n]

    def to_t(a):
        return torch.tensor(a).unsqueeze(-1).to(device)

    X_tr_t  = to_t(X_tr);  y_tr_t  = torch.tensor(y_tr).unsqueeze(-1).to(device)
    X_val_t = to_t(X_val); y_val_t = torch.tensor(y_val).unsqueeze(-1).to(device)
    X_te_t  = to_t(X_te)

    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                    batch_size=batch_size, shuffle=True)

    model    = _LSTMNet().to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn  = nn.MSELoss()
    best_val, best_state = float("inf"), None
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr_t)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={ep_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping на эпохе {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_te_t).cpu().numpy().flatten()

    pred       = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred       = np.clip(pred, 0, None)
    y_test_out = scaler.inverse_transform(y_seq[split:].reshape(-1, 1)).flatten()

    if plot_loss:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train Loss", color=PALETTE[0])
        ax.plot(val_losses,   label="Val Loss",   color=PALETTE[1])
        ax.set_title("LSTM — Loss в процессе обучения")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig("lstm_loss.png", dpi=120)
        plt.show()

    print("✓ LSTM — прогноз готов")
    return pred, y_test_out