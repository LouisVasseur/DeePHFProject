from datasets import Dataset
import click

seed = 42


@click.command()

@click.option(
    '--dataset-path',
    '-d',
    required=True
)

@click.option(
    "--split-path",
    '-s',
    required=True
)

def main (dataset_path: str, split_path: str):

    dataset = Dataset.load_from_disk(dataset_path)

    split_1 = dataset.train_test_split(test_size=0.2, seed=seed)
    split_2 = split_1["test"].train_test_split(test_size = 0.5, seed=seed)

    split_1["train"].save_to_disk(f'{split_path}/train')
    split_2["train"].save_to_disk(f'{split_path}/val')
    split_2["test"].save_to_disk(f'{split_path}/test')

if __name__=="__main__":
    main()