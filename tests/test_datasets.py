import pytest
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import DeepONetDataset, DNNDataset, PINNDataset


@pytest.mark.parametrize("batch_size", [32, 128])
def test_datasets(batch_size):
    kwds = dict(
        data_dir="data",
        filename="Burgers_2048_0.1",
        num_data=5000,
        idx=0,
        num_input_sensors=128,
        num_output_sensors=100,
        batch_size=batch_size,
    )

    def test_dataset(dataset: Dataset, *args, **kwargs) -> None:
        dataset: Dataset = dataset(*args, **kwargs)
        loader: DataLoader = DataLoader(dataset, batch_size)
        data, target = next(iter(loader))
        print("Dataset:", dataset.__class__.__name__)
        if isinstance(data, list):
            print(data[0].size(), data[1].size(), target.size())
        else:
            print(data.size(), target.size())

    test_dataset(DNNDataset, **kwds)
    test_dataset(PINNDataset, **kwds)
    test_dataset(DeepONetDataset, **kwds)
