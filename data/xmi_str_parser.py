"""Parse AutoVB ``.xmi`` + ``.str`` inputs into ``ParsedXMOMolecule`` facts."""

from __future__ import annotations

from pathlib import Path
import re

from data.xmo_parser import ParsedVBStructure, ParsedXMOMolecule, XmoParser


class XmiStrParser:
    """
    Build one parsed molecule object from paired ``.xmi`` and ``.str`` files.

    The current training pipeline expects the same typed facts produced by
    ``XmoParser``. ``.xmi`` already contains the shared static molecule fields
    (``$ctrl``, ``$geo``, ``$orb``), while ``.str`` provides the VB-structure
    list needed for structure-level graph construction.
    """

    def __init__(self, xmi_path: str | Path, str_path: str | Path):
        self.xmi_path = Path(xmi_path)
        self.str_path = Path(str_path)
        self.xmi_parser = XmoParser(self.xmi_path)

    def readStructureLines(self) -> list[str]:
        """Read non-empty lines from the ``.str`` file."""

        text = self.str_path.read_text(encoding="utf-8", errors="ignore")
        return [line.rstrip("\n") for line in text.splitlines() if line.strip()]

    def parseStructures(self) -> list[ParsedVBStructure]:
        """
        Parse VB structures from ``.str``.

        Expected line shape:
        ``index  *****  1:k  active_orbital_tokens...``

        Unlike ``.xmo``, ``.str`` does not carry Lowdin weights. For inference
        we attach a dummy placeholder weight of ``0.0`` because the downstream
        builder only needs structure strings and a deterministic ordering.
        """

        structures: list[ParsedVBStructure] = []
        pattern = re.compile(r"^\s*(\d+)\s+\*{2,}\s+(.*)$")

        for line in self.readStructureLines():
            match = pattern.match(line)
            if match is None:
                continue

            vb_index = int(match.group(1)) - 1
            structure_string = match.group(2).strip()
            structures.append(
                ParsedVBStructure(
                    vb_index=vb_index,
                    structure_string=structure_string,
                    lowdin_weight=0.0,
                    orb_pairs=self.xmi_parser.parseActivePairs(structure_string),
                )
            )

        structures.sort(key=lambda structure: structure.vb_index)
        if len(structures) == 0:
            raise ValueError(f"Cannot parse any VB structures from {self.str_path}")
        return structures

    def parse(self) -> ParsedXMOMolecule:
        """Parse one paired ``.xmi`` + ``.str`` inference input."""

        text = self.xmi_parser.readText()
        nao, nae = self.xmi_parser.parseCtrlCounts(text)
        geo_lines = self.xmi_parser.collectBlockLines(text, "geo", skip_first_data_line=False)
        orb_lines = self.xmi_parser.collectBlockLines(text, "orb", skip_first_data_line=True)
        atom_symbols, atom_numbers, atom_positions = self.xmi_parser.parseGeo(geo_lines)
        orb2atom, orb_role_id, atom_from_orb = self.xmi_parser.parseOrbBlock(
            orb_lines,
            nao,
            atom_numbers,
        )
        structures = self.parseStructures()

        molecule_stem = self.xmi_path.stem
        if molecule_stem.endswith("_vb"):
            molecule_stem = molecule_stem[:-3]

        return ParsedXMOMolecule(
            molecule_id=molecule_stem,
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
