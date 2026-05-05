"""Targeted parser tests for XMO active-pair decoding."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.xmo_parser import XmoParser


def makeParser() -> XmoParser:
    return XmoParser("dummy.xmo")


def testParseActivePairsSupportsSpaceSeparatedPairs() -> None:
    parser = makeParser()

    assert parser.parseActivePairs("1:23 26 27 25 28 24 29") == [
        (25, 26),
        (24, 27),
        (23, 28),
    ]


def testParseActivePairsSupportsHyphenSeparatedPairs() -> None:
    parser = makeParser()

    assert parser.parseActivePairs("1:22 25-26 24-27 23-28") == [
        (24, 25),
        (23, 26),
        (22, 27),
    ]


def testParseActivePairsSupportsMixedRepeatedAndHyphenTokens() -> None:
    parser = makeParser()

    assert parser.parseActivePairs("1:22 23 23 25-26 24-27") == [
        (22, 22),
        (24, 25),
        (23, 26),
    ]


def testParseActivePairsDoesNotTreatPrefixAsActivePair() -> None:
    parser = makeParser()

    assert parser.parseActivePairs("1:23 26 27 25 28 24 29") == [
        (25, 26),
        (24, 27),
        (23, 28),
    ]
    assert parser.parseActivePairs("1:22 25-26 24-27 23-28") == [
        (24, 25),
        (23, 26),
        (22, 27),
    ]
    assert parser.parseActivePairs("1:22 23 23 25-26 24-27") != [
        (21, 21),
        (22, 22),
        (24, 25),
        (23, 26),
    ]


def testParseLowdinWeightsSupportsHyphenSeparatedStructures() -> None:
    parser = makeParser()
    text = """
         Lowdin Weights

           1       0.00516  ******   1:28   32-33    31-34    30-35    29-36
           2       0.00595  ******   1:28   31-32    33-34    30-35    29-36
         Inverse Weights
    """

    structures = parser.parseLowdinWeights(text)

    assert [structure.vb_index for structure in structures] == [0, 1]
    assert [structure.lowdin_weight for structure in structures] == [0.00516, 0.00595]
    assert structures[0].orb_pairs == [(31, 32), (30, 33), (29, 34), (28, 35)]


def testParseLowdinWeightsSupportsSpaceSeparatedStructures() -> None:
    parser = makeParser()
    text = """
         Lowdin Weights

       1     0.00456  ******  1:28  32  33  31  34  30  35  29  36
       2     0.00575  ******  1:28  31  32  33  34  30  35  29  36
         Inverse Weights
    """

    structures = parser.parseLowdinWeights(text)

    assert [structure.vb_index for structure in structures] == [0, 1]
    assert [structure.lowdin_weight for structure in structures] == [0.00456, 0.00575]
    assert structures[0].orb_pairs == [(31, 32), (30, 33), (29, 34), (28, 35)]
