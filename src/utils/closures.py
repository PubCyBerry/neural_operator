from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset

grad_kwargs: dict = {"create_graph": True, "retain_graph": True}


class Closure:
    def __init__(
        self, model_name: str, dataset: Optional[Dataset], *args: Any, **kwds: Any
    ) -> None:
        """
        model_name: key for closure
        coefficient: property of PDE instance
        dataset: (PINN) Hold collocation data from dataset
        """
        self.coefficient: Optional[torch.tensor] = torch.Tensor(dataset.coefficient)

        model_name: str = model_name.lower()
        if model_name == "dnn":
            self.closure = self.data_loss

        elif model_name == "pinn":
            self.ic_data = dataset.ic_data
            self.bc_data = dataset.bc_data
            self.col_data = dataset.col_data
            self.closure = self.pinn_loss

        elif model_name == "deeponet":
            self.closure = self.data_loss

        elif model_name == "fno":
            return NotImplementedError

        elif model_name == "fno-time":
            return NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.closure(*args, **kwds)

    def data_loss(
        self,
        model: torch.nn.Module,
        data: Tuple[torch.tensor, torch.tensor],
        target: torch.tensor,
        device: torch.device,
    ) -> torch.tensor:
        # Data-driven Loss
        preds = model(data)
        loss = F.mse_loss(preds, target)
        return loss

    def pinn_loss(
        self,
        model: torch.nn.Module,
        data: Tuple[torch.tensor, torch.tensor],
        target: torch.tensor,
        device: torch.device,
    ) -> torch.tensor:
        """
        Sum up all losses
        loss_d: data-driven loss
        loss_ic: initial condition loss
        loss_bc: boundary condition loss
        loss_g: governing equation loss
        """
        loss_d = self.data_loss(model, data, target, device)
        loss_ic = self.ic_loss(model, *self.ic_data.push_data(), device) 
        loss_bc = self.bc_loss(model, *self.bc_data.push_data(), device)
        loss_g = self.collocation_loss(model, self.col_data.push_data(), device)

        loss = loss_d + loss_ic + loss_bc + loss_g
        return loss

    def ic_loss(
        self,
        model: torch.nn.Module,
        data: Tuple[torch.tensor, torch.tensor],
        target: torch.tensor,
        device: torch.device,
    ) -> torch.tensor:
        # Enforce model follow initial condition
        preds = model(data)
        loss = F.mse_loss(preds, target) 
        return loss / len(preds)

    def bc_loss(
        self,
        model: torch.nn.Module,
        left_data: Tuple[torch.tensor, torch.tensor],
        right_data: Tuple[torch.tensor, torch.tensor],
        device: torch.device,
    ) -> torch.tensor:
        # Enforce model follow (periodic) boundary condition
        left_preds = model(left_data)
        right_preds = model(right_data)
        loss = F.mse_loss(left_preds, right_preds)
        return loss / (len(left_preds) * len(right_preds))

    def collocation_loss(
        self, model: torch.nn.Module, data: Tuple[torch.tensor, torch.tensor], device: torch.device
    ) -> torch.tensor:
        # Set variables differentiable
        x = data[0].requires_grad_()
        t = data[1].requires_grad_()
        u = model((x, t))
        # Compute residual & loss
        residual = self.compute_Burgers(u, x, t)
        pde_loss = F.mse_loss(residual, torch.zeros_like(residual))
        return pde_loss / len(residual)

    def compute_Burgers(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.tensor:
        # PDE Loss
        # u_t + uu_x = \nu u_xx(\nu = 0.1 in this dataset)
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), **grad_kwargs)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), **grad_kwargs)[0]
        u_xx = grad(u, x, grad_outputs=torch.ones_like(u), **grad_kwargs)[0]
        return u_t + u * u_x - self.coefficient * u_xx
