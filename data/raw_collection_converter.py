"""Convert legacy torch-saved raw collection into native pickle format."""

import argparse
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List

import numpy as np


class RawCollectionConverter:
    """
    Convert ``raw/vb_data_collection.pt`` into a torch-free native pickle file.

    Purpose:
    - Keep dataset semantics unchanged.
    - Remove runtime dependency on ``torch.load`` for training and testing.
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Initialize converter.

        Arguments:
        - input_path: Path to legacy ``.pt`` raw collection.
        - output_path: Path to converted native ``.pkl`` collection.
        """

        self.input_path = input_path
        self.output_path = output_path
        self.project_root = str(Path(__file__).resolve().parents[1])

    def toPython(self, value: Any):
        """
        Convert tensor-like values into python lists recursively.

        Arguments:
        - value: Any tensor/list/tuple/scalar object.

        Returns:
        - Converted python object.
        """

        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
            return value.detach().cpu().tolist()
        if isinstance(value, tuple):
            return [self.toPython(item) for item in value]
        if isinstance(value, list):
            return [self.toPython(item) for item in value]
        return value

    def numpyArray(self, value: Any, dtype) -> np.ndarray:
        """
        Convert tensor-like value into numpy array with target dtype.

        Arguments:
        - value: Tensor-like or array-like value.
        - dtype: Target numpy dtype.

        Returns:
        - Numpy array.
        """

        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.detach().cpu().numpy().astype(dtype)
        return np.asarray(value, dtype=dtype)

    def readLegacyCollection(self) -> Dict[str, Any]:
        """
        Load legacy collection from torch archive.

        Returns:
        - Raw collection dictionary.
        """

        import torch
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

        try:
            return torch.load(
                self.input_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                self.input_path,
                map_location="cpu",
            )

    def convertMolecule(self, molecule) -> Dict[str, Any]:
        """
        Convert one legacy molecule object into native dictionary.

        Arguments:
        - molecule: Legacy ``VBinformation`` object.

        Returns:
        - Native dictionary with python/numpy fields.
        """

        return {
            "molecule_id": str(molecule.molecule_id),
            "nodes": int(molecule.nodes),
            "nao": int(molecule.nao),
            "nae": int(molecule.nae),
            "sym": [str(x) for x in molecule.sym],
            "atom_nums": self.numpyArray(molecule.atom_nums, np.int32),
            "coor": self.numpyArray(molecule.coor, np.float32),
            "str": self.toPython(molecule.str),
            "atom_from_orb": self.toPython(molecule.atom_from_orb),
            "A_mat": self.numpyArray(molecule.A_mat, np.int32),
            "A_list": self.toPython(molecule.A_list),
            "E": self.toPython(molecule.E),
            "X": self.numpyArray(molecule.X, np.float32),
            "LowdinWeights": self.toPython(molecule.LowdinWeights),
        }

    def run(self) -> None:
        """
        Execute raw collection conversion and save native pickle.

        Returns:
        - None.
        """

        legacy = self.readLegacyCollection()
        converted_molecules: List[Dict[str, Any]] = [
            self.convertMolecule(molecule) for molecule in legacy["molecules"]
        ]
        payload = {
            "molecules": converted_molecules,
            "num_molecules": int(legacy.get("num_molecules", len(converted_molecules))),
        }
        with open(self.output_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        print(f"Converted molecules: {len(converted_molecules)}")
        print(f"Output file: {self.output_path}")


def main() -> None:
    """
    Parse CLI arguments and run conversion.

    Returns:
    - None.
    """

    parser = argparse.ArgumentParser(description="Convert legacy VB raw collection to native pickle.")
    parser.add_argument("--input", type=str, required=True, help="Path to legacy raw .pt file.")
    parser.add_argument("--output", type=str, required=True, help="Path to native output .pkl file.")
    args = parser.parse_args()

    converter = RawCollectionConverter(input_path=args.input, output_path=args.output)
    converter.run()


if __name__ == "__main__":
    main()
