from typing import Any, Optional

import torch
from torchmetrics import Metric


# loss function with rel/abs Lp loss
class LpLoss(Metric):
    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        size_average: bool = True,
        reduction: bool = True,
        *args: Any,
        **kwds: Any,
    ) -> None:
        super().__init__()
        self.add_state("lploss", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.Tensor([0]), dist_reduce_fx="sum")

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

        # Set to True if the metric reaches it optimal value when the metric is maximized.
        # Set to False if it when the metric is minimized.
        higher_is_better: Optional[bool] = False

    def update(self, preds: torch.tensor, target: torch.tensor):
        self.lploss += self.rel(preds, target)
        self.total += len(target)

    def compute(self):
        return self.lploss / self.total

    def abs(self, x: torch.tensor, y: torch.tensor):
        """Absolute Lp Loss."""
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x: torch.tensor, y: torch.tensor):
        """Relative Lp Loss."""
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(Metric):
    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        k: int = 1,
        a: Any = None,
        group: bool = False,
        size_average: bool = True,
        reduction: bool = True,
        *args: Any,
        **kwds: Any,
    ):
        super().__init__()
        self.add_state("hsloss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a is None:
            a = [1] * k
        self.a = a

        # Set to True if the metric reaches it optimal value when the metric is maximized.
        # Set to False if it when the metric is minimized.
        higher_is_better: Optional[bool] = False

    def update(self, preds: torch.tensor, target: torch.tensor):
        self.hsloss += self.compute_norm(preds, target)
        self.total += len(target)

    def compute(self):
        return self.hsloss / self.total

    def rel(self, x: torch.tensor, y: torch.tensor):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def compute_norm(self, x: torch.tensor, y: torch.tensor, a: Any = None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1),
                    torch.arange(start=-nx // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=ny // 2, step=1),
                    torch.arange(start=-ny // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
        )
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced is False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss
