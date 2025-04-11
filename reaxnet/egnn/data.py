"""Create a mapping of atomic numbers.
"""
import yaml
import numpy as np
from typing import Sequence
from collections import namedtuple
from dataclasses import dataclass
from matscipy.neighbours import neighbour_list
from typing import Optional, Tuple, Dict, List
from ase import Atoms
import jax
import warnings
from tqdm import tqdm
import jraph

class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        """
        The `AtomicNumberTable` class should be initialized with the set of all atomic numbers that will be used in the dataset.
        """
        zs = [int(z) for z in zs]
        # check uniqueness
        assert len(zs) == len(set(zs))
        # check sorted
        if not zs == sorted(zs):
            zs = sorted(zs)
        _type = np.arange(1, len(zs) + 1)
        _type = [int(t) for t in _type]
        self.mapping_type = dict(zip(zs, _type))
        self._map_fn = np.vectorize(lambda x: self.mapping_type[x])
        self.zs = zs
        
    def __len__(self) -> int:
        return len(self.zs)
    
    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)} Mapping: {self.mapping_type}"
    
    def mapping(self, atomic_numbers: np.ndarray) -> np.ndarray:
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        return self._map_fn(atomic_numbers)
    
    def save_mapping_dict(self, path: str='mapping.yml'):
        with open(path, 'w') as f:
            yaml.dump(self.mapping_type, f)
    
    @classmethod
    def from_dict(self, path: str):
        with open(path, 'r') as f:
            mapping_type = yaml.load(f, Loader=yaml.FullLoader)
        return AtomicNumberTable(mapping_type.keys())

def neighbor_list_fn(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    cell: Optional[np.ndarray] = None,  # [3, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert cell.shape == (3, 3)
    receivers, senders, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=[True, True, True],
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )
    return senders, receivers, unit_shifts


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: np.ndarray  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[np.ndarray] = None  # eV/Angstrom
    stress: Optional[np.ndarray] = None  # eV/Angstrom^3
    cell: Optional[np.ndarray] = None
    pbc: Optional[Tuple] = None
    senders: Optional[np.ndarray] = None
    receivers: Optional[np.ndarray] = None
    shifts: Optional[np.ndarray] = None
    weight: float = 1.0
    config_type: Optional[str] = 'Default'

Configurations = List[Configuration]

def config_from_atoms(
    atoms: Atoms,
    cutoff: float = 5.0,
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = {"Default": 1.0}
    try:
        energy = np.array([atoms.get_potential_energy()]) # eV
    except:
        warnings.warn("Could not get energy from ASE atoms")
        energy = np.array(0.0)
    try:
        forces = atoms.get_forces() # eV / Ang
    except:
        warnings.warn("Could not get forces from ASE atoms")
        forces = np.zeros((len(atoms), 3))    
    try:
        # We always use the stress in ASE format (negative of VASP / LAMMPS)
        stress = atoms.get_stress(voigt=False) # eV / Ang^3
    except:
        warnings.warn("Could not get stress from ASE atoms")
        stress = np.zeros((3, 3))
    atomic_numbers = atoms.numbers 
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    assert np.linalg.det(cell) >= 0.0
    config_type = atoms.info.get("config_type", "Default")
    weight = config_type_weights.get(config_type, 1.0)
    senders, receivers, shifts = neighbor_list_fn(atoms.get_positions(), cutoff=cutoff, cell=cell)
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.get_positions(), pbc=pbc, cell=cell,
        energy=energy, forces=forces, stress=stress, weight=weight,
        config_type=config_type, 
        senders=senders, receivers=receivers, shifts=shifts 
    )

def load_from_atomic_list(
    atoms_list: List[Atoms],
    cutoff: float = 5.0,
    config_type_weights: Dict = None,
) -> Tuple[Configurations, set[int], Dict[str, float]]: 
    configs, num_nodes, num_edges, energy = [], [], [], []
    for atoms in tqdm(atoms_list, desc="Loading configs from trajectory"):
        config = config_from_atoms(atoms, cutoff, config_type_weights)
        configs.append(config)
        num_nodes.append(len(atoms))
        num_edges.append(config.senders.shape[0])
        energy.append(config.energy)
    return configs

GraphNodes = namedtuple("Nodes", ["positions", "forces", "species"])
GraphEdges = namedtuple("Edges", ["shifts"])
GraphGlobals = namedtuple("Globals", ["cell", "energy", "stress", "weight"])

def graph_from_configuration(
    config: Configuration, z_table: AtomicNumberTable 
) -> jraph.GraphsTuple:
    if z_table is None:
        species = config.atomic_numbers
    else:
        species = z_table.mapping(config.atomic_numbers)

    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=config.positions,
            forces=config.forces,
            species=jax.nn.one_hot(species, len(z_table)+1),
        ),
        edges=GraphEdges(shifts=config.shifts),
        globals=jax.tree_util.tree_map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=config.cell,
                energy=config.energy,
                stress=config.stress,
                weight=np.asarray(config.weight),
            ),
        ),
        receivers=config.receivers,
        senders=config.senders,
        n_edge=np.array([config.senders.shape[0]]),
        n_node=np.array([config.positions.shape[0]]),
    )