from deephf_dataloader import load_dataset
from datasets import Dataset
import os
import click 

def get_hf_dataset (name_dataset: str, data_dir_path: str, save_path: str):

    """
    This function coverts a given partition of MOB-ML into an HF dataset
    """

    mob_ml_dataset = load_dataset(
        name = name_dataset,
        data_dir = data_dir_path
    )

    # Converts to HF dataset
    mob_ml_dataset_hf = Dataset.from_dict({
        'mol_id' : [molecule.mol_id for molecule in mob_ml_dataset],
        'E_HF_hartree' : [molecule.properties['E_HF_hartree'] for molecule in mob_ml_dataset],
        'E_MP2_hartree' : [molecule.properties['E_MP2_hartree'] for molecule in mob_ml_dataset],
        'E_CCSDT_hartree' : [molecule.properties['E_CCSDT_hartree'] for molecule in mob_ml_dataset],
        'E_corr_hartree' : [molecule.properties['E_corr_hartree'] for molecule in mob_ml_dataset],
        'E_corr_eV' : [molecule.properties['E_corr_eV'] for molecule in mob_ml_dataset],
        'E_corr_kcal' : [molecule.properties['E_corr_kcal'] for molecule in mob_ml_dataset],
        'xyz_path' : [molecule.properties['xyz_path'] for molecule in mob_ml_dataset]
    })

    # Makes directory to save data
    os.makedirs(save_path, exist_ok=True)

    # Save dataset
    mob_ml_dataset_hf.save_to_disk(save_path)

@click.command()
@click.option("--name_dataset", "-d", type=str, default = 'water')
@click.option("--data_dir_path", "-p", type=str)
@click.option("--save_path", '-s', type=str)

def main(
    name_dataset: str,
    data_dir_path: str,
    save_path: str
):
    get_hf_dataset(name_dataset,
                   data_dir_path,
                   save_path)
    
if __name__ == "__main__":
    main()



