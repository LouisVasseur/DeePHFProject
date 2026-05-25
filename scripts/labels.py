"""Shared labels and colours used by figures.py and tables.py."""

DATASETS = ("water", "alkanes", "qm7b_T", "gdb13_T")
DATASET_LABEL = {"water": "Water", "alkanes": "Alkanes",
                 "qm7b_T": "QM7b-T", "gdb13_T": "GDB-13-T"}

ARCHITECTURES = ("A_mlp", "B_mpnn", "C_gat", "D_gat_edge")
ARCH_LABEL = {"A_mlp": "A: MLP", "B_mpnn": "B: MPNN",
              "C_gat": "C: GAT", "D_gat_edge": "D: GAT-edge"}

DESCRIPTORS = ("chemical", "chemical_elec", "chemical_soap", "chemical_elec_soap")
DESC_LABEL = {"chemical": "chem", "chemical_elec": "chem+elec",
              "chemical_soap": "chem+SOAP",
              "chemical_elec_soap": "chem+elec+SOAP"}
DESC_COLOR = {"chemical": "#bbbbbb", "chemical_elec": "#4f7fb4",
              "chemical_soap": "#d18b3e", "chemical_elec_soap": "#5e9e60"}

CHEMICAL_ACCURACY = 1.594   # mHa, = 1 kcal/mol


def by_key(records, primary_seed=43):
    """{(dataset, descriptor, architecture): test_mae_mHa} for the primary seed,
    falling back to any other seed if missing."""
    primary, fallback = {}, {}
    for r in records:
        if r.get("test_mae_mHa") is None:
            continue
        key = (r["dataset"], r["descriptor"], r["architecture"])
        if r.get("seed") == primary_seed:
            primary[key] = r["test_mae_mHa"]
        else:
            fallback.setdefault(key, r["test_mae_mHa"])
    return {**fallback, **primary}
