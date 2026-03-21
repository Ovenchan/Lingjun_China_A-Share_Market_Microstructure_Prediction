import torch
import torch.nn as nn


class MoEGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...] = (32, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = (input_dim,) + tuple(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_experts))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


class FrozenExpertMoE(nn.Module):
    def __init__(
        self,
        gate_input_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...] = (32, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = MoEGate(
            input_dim=gate_input_dim,
            num_experts=num_experts,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        gate_inputs: torch.Tensor,
        expert_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_predictions.dim() != 2:
            raise ValueError("expert_predictions must have shape [batch, num_experts].")

        weights = self.gate(gate_inputs)
        prediction = torch.sum(weights * expert_predictions, dim=-1)
        return prediction, weights
