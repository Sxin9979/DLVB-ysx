from dataclasses import dataclass, field
from typing import List

@dataclass
class VBinformation:
    molecule_id: str = "" # 分子标识符（使用文件名）
    nodes: int = 0
    nao: int=0
    nae: int=0
    sym: List[str] = field(default_factory=list)  # 原子符号
    atom_nums: List[int] = field(default_factory=list) # 原子序数
    coor: List = field(default_factory=list) # 笛卡尔坐标
    str: List = field(default_factory=list) # VB结构
    atom_from_orb: List = field(default_factory=list) 
    A_mat: List = field(default_factory=list) # 邻接矩阵
    A_list: List[List[int]] = field(default_factory=lambda: [[], []]) # 邻接列表
    E: List = field(default_factory=list) # 边特征，包括活性电子数、非活性电子数以及边的相对坐标
    X: List = field(default_factory=list) # 节点特征
    LowdinWeights: List = field(default_factory=list) # lowdin权重

# class VBinformation:
#     def __init__(self):
#         self.molecule_id = ""  # 分子标识符（使用文件名）
#         self.nodes = 0
#         self.sym = []  # 原子符号
#         self.atom_nums = []
#         self.coor = []
#         self.str = []
#         self.atom_from_orb = []
#         self.A_mat = []  # 邻接矩阵
#         self.A_list = [[], []]  # 邻接列表
#         self.E = []  # 边特征，包括活性电子数、非活性电子数以及边的相对坐标
#         self.X = []  # 节点特征
#         self.LowdinWeights = []