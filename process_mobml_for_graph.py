from datasets import Dataset
from graph_atomic_descriptors import GraphAtomicDescriptors
import click
import logging

graph_encoder = GraphAtomicDescriptors(remove_h=False)

def filter_valid_entries (datapoint: dict) -> bool:
    """
    Function to filter the datapoints that don't have a valid graph representation

    Inputs:
        datapoint (dict) : Datapoint from an HF dataset that contains
            the graph representation of the molecule
    Output:
        True if point in batch is valid, False otherwise
    """

    not_removed = (
        datapoint["node_feature_matrix"] is not None and
        datapoint["edge_feature_matrix"] is not None and
        datapoint["indices_feature_matrix"] is not None
    )
    if not(not_removed):
        logging.warning(f"Datapoint at {datapoint['mol_id']} is removed due to invalid graph representation")

    return not_removed


def encode_molecule(batch: dict) -> dict:
    """
    Bathced function to encode datapoints into graph

    Inputs:
        batch (dict) : A batch from an HF dataset that contains
            the key 'xyz' path
    Output:
        results (dict) : A dictionary that encoded forms of each datapoint
    """

    results = {"node_feature_matrix": [],
               "edge_feature_matrix": [],
               "indices_feature_matrix": []}

    for idx, xyz_path in enumerate(batch["xyz_path"]):

        try:
            node, edge, indices = graph_encoder.build_molecular_graph(xyz_path=xyz_path)
        except Exception as e:
            logging.warning(f"Could not process entry {batch[idx]['mol_id']} due to {e}; datapoint will be deleted")
            node, edge, indices = None, None, None

        results["node_feature_matrix"].append(node)
        results["edge_feature_matrix"].append(edge)
        results["indices_feature_matrix"].append(indices)

    return results


@click.command()

@click.option(
    "--dataset-path",
    "-p",
    required=True,
    help = "Path to the dataset that needs processing for GNN"
)
@click.option(
    "--log-file",
    "-l",
    required = True,
    help = "File that contains the datapoints that could not be processed"
)
@click.option(
    "--output-path",
    "-o",
    required = True,
    help = "Directory where the processed dataset will be saved"
)

def main (dataset_path: str, log_file:str, output_path: str):

    logging.basicConfig(
        filename=log_file,
        level = logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    dataset = Dataset.load_from_disk(dataset_path)
    dataset = dataset.map(encode_molecule, batch_size = 64, batched=True)
    dataset = dataset.filter(filter_valid_entries)
    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    main()