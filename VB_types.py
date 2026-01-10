class VBinformation:
    def __init__(self):
        self.molecule_id = ""  # 分子标识符（使用文件名）
        self.nodes = 0
        self.sym = []  # 原子符号
        self.atom_nums = []
        self.coor = []
        self.str = []
        self.atom_from_orb = []
        self.A_mat = []  # 邻接矩阵
        self.A_list = [[], []]  # 邻接列表
        self.E = []  # 边特征，包括活性电子数、非活性电子数以及边的相对坐标
        self.X = []  # 节点特征
        self.LowdinWeights = []