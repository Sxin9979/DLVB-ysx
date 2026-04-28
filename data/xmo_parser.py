"""Parse XMVB ``.xmo`` files into structured molecule-level facts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


ATOM_TO_NUMBER = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
}


@dataclass
class ParsedVBStructure:
    """One VB structure parsed from the Lowdin-weights section."""

    vb_index: int
    structure_string: str
    lowdin_weight: float
    orb_pairs: list[tuple[int, int]]


@dataclass
class ParsedXMOMolecule:
    """Structured facts extracted from one ``.xmo`` file."""

    molecule_id: str
    atom_symbols: list[str]
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    nao: int
    nae: int
    orb2atom: list[list[int]]
    orb_role_id: np.ndarray
    atom_from_orb: np.ndarray
    structures: list[ParsedVBStructure]


class XmoParser:
    """Parse one ``.xmo`` file into geometry, orbital, and VB-structure facts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def readText(self) -> str:
        """Read text with a permissive encoding fallback."""

        return self.path.read_text(encoding="utf-8", errors="ignore")

    def parseCtrlCounts(self, text: str) -> tuple[int, int]:
        """Extract ``nao`` and ``nae`` from the ``$ctrl`` block."""

        match = re.search(r"(?is)\$ctrl(.*?)(\$end|\n\s*\$)", text)
        block = match.group(1) if match else text
        nao_match = re.search(r"(?i)\bnao\s*=\s*(\d+)\b", block)
        nae_match = re.search(r"(?i)\bnae\s*=\s*(\d+)\b", block)
        if (nao_match is None) or (nae_match is None):
            raise ValueError(f"Cannot parse nao/nae from {self.path}")
        return int(nao_match.group(1)), int(nae_match.group(1))

    def collectBlockLines(self, text: str, block_name: str, skip_first_data_line: bool = False) -> list[str]:
        """Collect non-empty lines inside one ``$block ... $end`` region."""

        in_block = False
        skip_next = bool(skip_first_data_line)
        block_lines: list[str] = []

        for line in text.splitlines():
            line_low = line.lower().strip()
            if re.match(rf"^\$[\s]*{re.escape(block_name.lower())}\b", line_low):
                in_block = True
                skip_next = bool(skip_first_data_line)
                block_lines = []
                continue

            if not in_block:
                continue

            if re.match(r"^\$[\s]*end\b", line_low):
                break

            if skip_next:
                skip_next = False
                continue

            if line.strip():
                block_lines.append(line.rstrip("\n"))

        if len(block_lines) == 0:
            raise ValueError(f"Cannot find valid ${block_name} block in {self.path}")
        return block_lines

    def parseGeo(self, geo_lines: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Parse ``$geo`` lines into symbols, atomic numbers, and coordinates."""

        atom_symbols: list[str] = []
        atom_numbers: list[int] = []
        atom_positions: list[list[float]] = []

        for line in geo_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            symbol = parts[0].strip().capitalize()
            if symbol not in ATOM_TO_NUMBER:
                raise ValueError(f"Unsupported atom symbol {symbol} in {self.path}")
            atom_symbols.append(symbol)
            atom_numbers.append(ATOM_TO_NUMBER[symbol])
            atom_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return (
            atom_symbols,
            np.asarray(atom_numbers, dtype=np.int32),
            np.asarray(atom_positions, dtype=np.float32),
        )

    def parseOrbBlock(self, orb_lines: list[str], nao: int, atom_numbers: np.ndarray) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Parse the shared ``$orb`` block into orbital ownership metadata."""

        total_orbitals = len(orb_lines)
        active_start = total_orbitals - nao
        if active_start < 0:
            raise ValueError(
                f"Invalid orbital block in {self.path}: total_orbitals={total_orbitals}, nao={nao}"
            )

        orb2atom: list[list[int]] = []
        orb_role_id: list[int] = []
        atom_from_orb: list[list[int]] = []

        for orb_index, line in enumerate(orb_lines):
            line_clean = line.split("#")[0].strip()
            if not line_clean:
                raise ValueError(f"Empty orbital line at index {orb_index} in {self.path}")

            atom_ids = [int(token) - 1 for token in line_clean.split()]
            if orb_index < active_start:
                if len(atom_ids) == 1:
                    role = 0
                elif len(atom_ids) == 2:
                    role = 1
                else:
                    raise ValueError(f"Unexpected inactive orbital line: {line}")
            else:
                if len(atom_ids) != 1:
                    raise ValueError(f"Active orbital line should contain one atom index: {line}")
                role = 2

            if any((atom_id < 0) or (atom_id >= atom_numbers.shape[0]) for atom_id in atom_ids):
                raise ValueError(f"Orbital atom index out of range in {self.path}: {line}")

            representative_atom = atom_ids[0]
            orb2atom.append(atom_ids)
            orb_role_id.append(role)
            atom_from_orb.append(
                [
                    orb_index + 1,
                    int(atom_numbers[representative_atom]),
                    representative_atom + 1,
                ]
            )

        return (
            orb2atom,
            np.asarray(orb_role_id, dtype=np.int32),
            np.asarray(atom_from_orb, dtype=np.int32),
        )

    def parseActivePairs(self, structure_string: str) -> list[tuple[int, int]]:
        """Parse active-orbital pairings from one VB-structure string."""

        tokens = structure_string.strip().split()
        if len(tokens) <= 1:
            return []

        active_tokens = tokens[1:]
        active_orbitals: list[int] = []
        for token in active_tokens:
            token = token.strip()
            if not token:
                continue
            try:
                active_orbitals.append(int(token) - 1)
            except ValueError:
                continue

        if len(active_orbitals) % 2 == 1:
            active_orbitals = active_orbitals[:-1]

        return [
            (active_orbitals[index], active_orbitals[index + 1])
            for index in range(0, len(active_orbitals), 2)
        ]

    def parseLowdinWeights(self, text: str) -> list[ParsedVBStructure]:
        """Parse all VB structures and their Lowdin weights."""

        structures: list[ParsedVBStructure] = []
        in_lowdin = False

        for line in text.splitlines():
            line_strip = line.strip()
            if re.search(r"(?i)lowdin\s+weights", line_strip):
                in_lowdin = True
                continue

            if not in_lowdin:
                continue

            if re.search(r"(?i)inverse\s+weights", line_strip):
                break

            if not line_strip:
                continue

            match = re.match(
                r"^\s*(\d+)\s+([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)\s+\*{2,}\s+(.*)$",
                line,
            )
            if match is None:
                continue

            vb_index = int(match.group(1)) - 1
            lowdin_weight = float(match.group(2))
            structure_string = match.group(3).strip()
            structures.append(
                ParsedVBStructure(
                    vb_index=vb_index,
                    structure_string=structure_string,
                    lowdin_weight=lowdin_weight,
                    orb_pairs=self.parseActivePairs(structure_string),
                )
            )

        structures.sort(key=lambda structure: structure.vb_index)
        if len(structures) == 0:
            raise ValueError(f"Cannot find Lowdin weights in {self.path}")
        return structures

    def parse(self) -> ParsedXMOMolecule:
        """Parse one ``.xmo`` file into a typed molecule fact object."""

        text = self.readText()
        nao, nae = self.parseCtrlCounts(text)
        geo_lines = self.collectBlockLines(text, "geo", skip_first_data_line=False)
        orb_lines = self.collectBlockLines(text, "orb", skip_first_data_line=True)
        atom_symbols, atom_numbers, atom_positions = self.parseGeo(geo_lines)
        orb2atom, orb_role_id, atom_from_orb = self.parseOrbBlock(orb_lines, nao, atom_numbers)
        structures = self.parseLowdinWeights(text)

        return ParsedXMOMolecule(
            molecule_id=self.path.stem,
            atom_symbols=atom_symbols,
            atom_numbers=atom_numbers,
            atom_positions=atom_positions,
            nao=nao,
            nae=nae,
            orb2atom=orb2atom,
            orb_role_id=orb_role_id,
            atom_from_orb=atom_from_orb,
            structures=structures,
        )
