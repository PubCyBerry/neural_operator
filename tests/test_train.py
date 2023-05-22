import pytest

from src.models.models import DNN, PINN, DeepONet, FNO

@pytest.mark.parametrize("batch_size", [32])
def test_dnn(batch_size):
    raise NotImplementedError
    
    
@pytest.mark.parametrize("batch_size", [32])
def test_pinn(batch_size):
    raise NotImplementedError

@pytest.mark.parametrize("batch_size", [32])
def test_deeponet(batch_size):
    raise NotImplementedError

@pytest.mark.parametrize("batch_size", [32])
def test_fno(batch_size):
    raise NotImplementedError