from rdkit import Chem
import numpy as np
from openbabel import pybel, openbabel

# Constants that will be used for the encoding in the graph
atom_one_hot_encoding = {
    'C' : [1,0,0,0,0,0],
    'Cl' : [0,1,0,0,0,0],
    'N' : [0,0,1,0,0,0],
    'O' : [0,0,0,1,0,0],
    'S' : [0,0,0,0,1,0],
    'H' : [0,0,0,0,0,1]
}

bond_type_one_hot_encoding = {
    'SINGLE' : [1,0,0,0],
    'DOUBLE' : [0,1,0,0],
    'TRIPLE' : [0,0,1,0],
    'AROMATIC' : [0,0,0,1]
}

hybridization_one_hot_encoding = {
    'SP3' : [1,0,0,0,0,0,0],
    'SP2' : [0, 1, 0, 0, 0, 0, 0],
    'SP' : [0, 0, 1, 0, 0, 0, 0],
    'SP3D2' : [0, 0, 0, 1, 0, 0, 0],
    'SP3D' : [0, 0, 0, 0, 1, 0, 0],
    'S' : [0, 0, 0, 0, 0, 1, 0],
    'UNKNOWN' : [0, 0, 0, 0, 0, 0, 1]
}

pauling_electronegativity = {  # Taken from wikipedia
    6 : 2.55,
    17: 3.16,
    7: 3.04,
    8: 3.44,
    16: 2.58, 
    1: 2.2
}


class GraphAtomicDescriptors():

    def __init__ (self, remove_h: bool = False):
        self.remove_h = remove_h

    def build_node_features (self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """
        This function takes an RdKit mol object and builds
        a matrix containing the features of each atom from it.
        The features contain:
        - first 6 coordinates: one hot encoding of atom
        (among C, Cl, N, O, S, H)
        - coordinate 7: atomic number
        - coordinate 8: is aromatic (1 or 0)
        - coordinate 9-15: one hot encoding of hybridization
        (among SP3, SP2, SP, SP3D2, SP3D, S, UNKNOWN)
        - coordinate 16: number of implicit H atoms
        - coordinate 17: Pauling electronegativity

        Input:
            mol (Chem.rdchem.Mol): RdKit molecule
        Output:
            node_feature_matrix (np.ndarray): Node feature matrix
                of dimensions (N_atoms, 17)
        """

        if type(mol) != Chem.rdchem.Mol:
            raise TypeError("Provide an RdKit Mol object")
        
        node_feature_matrix = []

        for atom in mol.GetAtoms():
            
            # Determine features
            identity_one_hot_encoding = atom_one_hot_encoding[atom.GetSymbol()]
            atomic_number = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hyb_one_hot_encoding = hybridization_one_hot_encoding[str(atom.GetHybridization())]
            h_number = atom.GetValence(Chem.ValenceType.IMPLICIT)
            electronegativity = pauling_electronegativity[atom.GetAtomicNum()]

            # Group the features into single array
            atom_features = [np.array(identity_one_hot_encoding, dtype=np.float32),
                            np.array([atomic_number], dtype=np.float32),
                            np.array([aromatic], dtype=np.float32),
                            np.array(hyb_one_hot_encoding, dtype=np.float32),
                            np.array([h_number], dtype=np.float32),
                            np.array([electronegativity], dtype=np.float32)]
            atom_features = np.concatenate(atom_features)

            # Append to the feature matrix
            node_feature_matrix.append(atom_features)
            
        # Build the feature matrix
        node_feature_matrix = np.array(node_feature_matrix, dtype=np.float32)

        return node_feature_matrix
        

    def build_edge_features (self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """
        This function takes an RdKit mol object and builds
        a matrix containing the features of each bond(between heavy atoms) from it.
        The features contain:
        - first 4 coordinates: one hot encoding of bond type
        (among SINGLE, DOUBLE, TRIPLE, AROMATIC)
        - coordinate 5: bond length
        
        Input:
            mol (Chem.rdchem.Mol): RdKit molecule
        Output:
            edge_feature_matrix (np.ndarray): Edge feature matrix
                of dimensions (2*N_bonds, 5)
        """

        if type(mol) != Chem.rdchem.Mol:
            raise TypeError("Provide an RdKit Mol object")
        conf = mol.GetConformer() # Needed for atom positions

        edge_feature_matrix = []

        for bond in mol.GetBonds():

            bond_type = bond_type_one_hot_encoding[str(bond.GetBondType())]
            
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            begin_atom_pos = np.array(conf.GetAtomPosition(begin_atom_idx))
            end_atom_pos = np.array(conf.GetAtomPosition(end_atom_idx))

            bond_length = np.linalg.norm(begin_atom_pos - end_atom_pos)

            bond_features = [np.array(bond_type, dtype=np.float32),
                            np.array([bond_length], dtype=np.float32)]
            
            edge_feature_matrix.append(np.concatenate(bond_features))

        edge_feature_matrix_unique_edges = np.array(edge_feature_matrix, dtype=np.float32)
        edge_feature_matrix = np.concat((edge_feature_matrix_unique_edges, edge_feature_matrix_unique_edges), axis=0)

        return edge_feature_matrix
    

    def build_edge_index_matrix (self, mol: Chem.rdchem.Mol)->np.ndarray:
        """
        This function takes an RdKit mol object and builds
        the edge indices matrix corresponding to the 
        undirected molecular graph.
        
        Input:
            mol (Chem.rdchem.Mol): RdKit molecule
        Output:
            edge_indices_matrix (np.ndarray): Edge indices matrix
                of the molecular graph of dimensions (2, 2*N_bonds)
        """

        edge_indices_direction_1 = []
        edge_indices_direction_2 = []

        for bond in mol.GetBonds():

            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            edge_indices_direction_1.append(np.array([begin_atom_idx, end_atom_idx], dtype=np.float32))
            edge_indices_direction_2.append(np.array([end_atom_idx, begin_atom_idx], dtype=np.float32))

        edge_indices_direction_1 = np.array(edge_indices_direction_1, dtype=np.float32).T
        edge_indices_direction_2 = np.array(edge_indices_direction_2, dtype=np.float32).T

        edge_indices_matrix = np.concat((edge_indices_direction_1, edge_indices_direction_2), axis=1)

        return edge_indices_matrix

    def build_molecular_graph (self, xyz_path):

        # Convert xyz file to sdf file in order to work with rdkit afterwards
        mol_original = next(pybel.readfile('xyz', xyz_path))
        sdf_string = mol_original.write('sdf')
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=self.remove_h)

        if mol is None:
            return (None, None, None)

        # Determine node features matrix
        node_feature_matrix = self.build_node_features(mol)

        # Determine edge feature matrix
        edge_feature_matrix = self.build_edge_features(mol)

        # Determine edge indices matrix
        edge_indices_matrix = self.build_edge_index_matrix(mol)

        return (node_feature_matrix, edge_feature_matrix, edge_indices_matrix)

    

