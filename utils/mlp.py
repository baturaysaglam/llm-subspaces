import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), nn.GELU()]
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        eval_ratio: float = 0.1,
        return_best_model: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        samples, labels = samples.to(device), labels.to(device)

        # shuffle & split
        idx = torch.randperm(samples.size(0))
        split = int(samples.size(0) * (1 - eval_ratio))
        X_train, y_train = samples[idx][:split], labels[idx][:split]
        X_eval,  y_eval  = samples[idx][split:], labels[idx][split:]

        loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=min(batch_size, len(X_train)), shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            nn.Module.train(self, True)
            for x, y in loader:
                opt.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                opt.step()

            nn.Module.train(self, False)
            with torch.no_grad():
                logits = self(X_eval)  # (N, 4)
                probs  = torch.softmax(logits, dim=1)
                preds = logits.argmax(1)
                acc = (preds == y_eval).float().mean().item()

                # ---------- binary grouping -------------
                group1_idx = torch.tensor([0, 2], device=probs.device)
                p_group1 = probs[:, group1_idx].sum(dim=1).cpu().numpy()  # score = P(group1)
                y_group = torch.isin(y_eval, group1_idx).long().cpu().numpy()  # 1 if label in {0,2}

                # ---------- metrics -------------------------
                g_auc = roc_auc_score(y_group, p_group1)  # grouped ROC-AUC
                acc = (preds == y_eval).float().mean().item()  # optional sanity-check

            print(f"epoch [{epoch}/{epochs}] | eval acc: {acc:.4f} | grouped AUC: {g_auc:.4f}")

        return self

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        logits = self.forward(samples)
        pred = logits.argmax(dim=-1)

        return pred
