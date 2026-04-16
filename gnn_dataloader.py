from datasets import Dataset
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def compute_stats(dataset: Dataset):
    ys = [ex["E_corr_eV"] for ex in dataset]
    y = torch.tensor(ys, dtype=torch.float)

    return {
        "mean": y.mean(),
        "std": y.std()
    }

def hf_to_pyg (datapoint: dict, stats: dict):

    x = torch.tensor(datapoint["node_feature_matrix"], dtype=torch.float)
    edge_index = torch.tensor(datapoint["indices_feature_matrix"], dtype=torch.long)
    edge_attr = torch.tensor(datapoint["edge_feature_matrix"], dtype=torch.float)

    y = torch.tensor([datapoint["E_corr_eV"]], dtype=torch.float)
    y = (y - stats["mean"]) / (stats["std"] + 1e-8)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

class hf_to_pyg_wrapper(TorchDataset):
    def __init__(self, hf_dataset, stats):
        self.ds = hf_dataset
        self.stats = stats

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return hf_to_pyg(self.ds[idx], self.stats)


def pyg_loader (dataset_path: str, shuffle : bool, batch_size: int, num_workers: int):

    # Load dataset and compute stats
    dataset = Dataset.load_from_disk (dataset_path)
    stats = compute_stats(dataset)

    pyg_dataset = hf_to_pyg_wrapper(dataset, stats)

    loaded_data = DataLoader(
        pyg_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers = num_workers
    )

    return loaded_data
