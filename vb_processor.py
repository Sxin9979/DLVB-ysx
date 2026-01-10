import torch
import os
import re
import numpy as np
from pathlib import Path
from ase import Atoms
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdDetermineBonds
from VB_types import VBinformation

class VBinfo_collector:
    def __init__(self):
        self.all_molecules=[] # 储存所有分子的VBinfo对象
    
    def atom_map(self): 
        """ 原子符号和原子序数的转换 """
        base_map = {
        'H': 1,     'He': 2,    'Li': 3,    'Be': 4,    'B': 5,
        'C': 6,     'N': 7,     'O': 8,     'F': 9,     'Ne': 10,
        'Na': 11,   'Mg': 12,   'Al': 13,   'Si': 14,   'P': 15,
        'S': 16,    'Cl': 17,   'Ar': 18,   'K': 19,    'Ca': 20,
        'Sc': 21,   'Ti': 22,   'V': 23,    'Cr': 24,   'Mn': 25,
        'Fe': 26,   'Co': 27,   'Ni': 28,   'Cu': 29,   'Zn': 30      
        }

        # 大小写转换
        def atom_to_number(atom_symbol):
            if atom_symbol in base_map:
                return base_map[atom_symbol]
            
            standardized = atom_symbol.capitalize()
            if standardized in base_map:
                return base_map[standardized]
            
            upper_case = atom_symbol.upper()
            if upper_case in base_map:
                return base_map[upper_case]
        
            print("元素对应有误")
            return 0

        return atom_to_number     
    
    # def covalent_radii(coor,nodes,atom_nums): 
    #     """ 生成共价半径和距离矩阵 """
    #     radius = {
    #     'H':0.31,   'He':0.28,  'Li':1.28,  'Be':0.96,  'B':0.84,
    #     'C':0.73,   'N':0.71,   'O':0.66,   'F':0.57,   'Ne':0.58,
    #     'Na':1.66,  'Mg':1.41,  'Al':1.21,  'Si':1.11,  'P':1.07,
    #     'S':1.05,   'Cl':1.02,  'Ar':1.06,  'K':2.03,   'Ca':1.76,
    #     'Sc':1.70,  'Ti':1.60,  'V':1.53,   'Cr':1.39,  'Mn':1.61,
    #     'Fe':1.52,  'Co':1.50,  'Ni':1.24,  'Cu':1.32,  'Zn':1.22     
    #     }
    #     radius_list=list(radius.values())

    #     c=1.2
    #     covalent_r=torch.zeros(nodes, nodes, dtype=torch.float32)
    #     distance_mat=torch.zeros(nodes,nodes,dtype=torch.float32)
    #     for i in range(nodes):
    #         for j in range(nodes):
    #             if i==j:
    #                 continue

    #             coor_1 = coor[i]
    #             coor_2 = coor[j] 
    #             distance_mat[i][j]=torch.norm(coor_1 - coor_2)
    #             covalent_r[i][j]=c*(radius_list[atom_nums[i]-1] + radius_list[atom_nums[j]-1]) 
    #             if i==0 and j==1:
    #                 print(atom_nums[i],atom_nums[j],radius_list[atom_nums[i]-1] ,radius_list[atom_nums[j]-1],distance_mat[0][1],covalent_r[i][j])            

    #     return covalent_r,distance_mat    
    
    def match_orbital_data(self, orbital_content): 
        """ 计算每条轨道上的原子系数之和 """

        all_orbitals = []
        current_block_orbitals = None
        num_orbitals = 0
        
        for line in orbital_content:
            line=line.strip() # 去除首尾空格

            # 检测轨道块头 (如: "1          2          3          4          5")
            # 规则：整行都是数字
            if re.match(r'^\d+(?:\s+\d+)*$', line):
                # 提取轨道编号
                orbital_numbers = re.findall(r'\d+', line) # 形如orbital_numbers=['1','2','3',...'20'] 
                num_orbitals = len(orbital_numbers) # 当前块的轨道数
                
                # 为当前块初始化轨道数据
                if current_block_orbitals is None:
                    current_block_orbitals = [defaultdict(float) for _ in range(num_orbitals)]
                else:
                    # 如果已经有轨道数据，先保存之前的块
                    all_orbitals.extend(current_block_orbitals)
                    current_block_orbitals = [defaultdict(float) for _ in range(num_orbitals)]
                continue
            
            # 处理数据行
            elif current_block_orbitals and re.match(r'^\s*\d+', line):
                parts=line.split()
                
                if len(parts) < 5:
                    continue     

                serial=parts[0]
                element=parts[1]
                atom_num=parts[2]
                orbital_type=parts[3]

                element_with_atom=f"{element}{atom_num}"
                
                for orbital_idx in range(num_orbitals): 
                    value_index = 4 + orbital_idx
                    if value_index < len(parts):
                        value=float(parts[value_index])
                        current_block_orbitals[orbital_idx][element_with_atom] += abs(value)
                    else:
                        break
            
        # 添加最后一个块的数据
        if current_block_orbitals:
            all_orbitals.extend(current_block_orbitals)
        
        # 转换为输出格式
        orb_sum = []
        for orbital_idx, orbital_data in enumerate(all_orbitals):
            # 按原子编号排序 (C1, C2, C3, C4, H5, H6, ...)
            sorted_atoms = sorted(
                orbital_data.items(), 
                key=lambda x: (x[0][0], int(x[0][1:]) if x[0][1:].isdigit() else 0)
            )
            orbital_list = [f"{atom}:{value:.6f}" for atom, value in sorted_atoms if value != 0]
            orb_sum.append(orbital_list)

        print("orb_sum:\n",orb_sum)
        return orb_sum

    def orb_match_atom(self, orbital_matched): 
        """ 根据每条轨道上的原子系数之和，将轨道与原子相配对 """
        atom_to_number = self.atom_map()
        result = []    

        for orb_idx, orb_data in enumerate(orbital_matched, 1):  # 轨道索引从1开始
            # 如果只有一个原子，直接取这个原子
            if len(orb_data) == 1:
                atom = orb_data[0].split(':')[0]
            else: # 如果有多个原子，找到值最大的那个
                max_item = max(orb_data, key=lambda x: float(x.split(':')[1]))
                atom = max_item.split(':')[0]
            
            # 解析原子符号和序号，分离字母部分和数字部分
            first_digit_idx = next((i for i, c in enumerate(atom) if c.isdigit()), len(atom))
            atom_symbol = atom[:first_digit_idx]
            atom_number = atom[first_digit_idx:]
            
            # 获取原子序数
            atomic_number = atom_to_number(atom_symbol) 
            atom_index = int(atom_number) if atom_number else 1
            
            result.append([orb_idx, atomic_number, atom_index])
        
        return torch.tensor(result, dtype=torch.long)

    # 通过RDkit得到成键方式
    def tensor_to_xyz_block(self, atoms, coords):
        """把元素和坐标tensor转换为XYZ格式字符串"""
        coords = coords.detach().cpu().numpy()
        
        lines = [f"{len(atoms)}", "Generated by RDKit"]
        for sym, (x, y, z) in zip(atoms, coords):
            lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
        return "\n".join(lines)

    def get_bond_info(self, atoms, coords):
        """根据原子坐标自动判断所有成键信息"""
        xyz_block = self.tensor_to_xyz_block(atoms, coords)
        mol = rdmolfiles.MolFromXYZBlock(xyz_block)
        if mol is None:
            raise ValueError("Failed to create molecule from XYZ")

        # 自动判断成键
        rdDetermineBonds.DetermineBonds(mol)

        # 提取成键信息
        bonds = []

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            idx1, idx2 = a1.GetIdx()+1, a2.GetIdx()+1
            symbol1, symbol2 = a1.GetSymbol(), a2.GetSymbol()
            order = str(bond.GetBondType())
            bd=0
            if order == "SINGLE":
                bd=1
            elif order == "DOUBLE":
                bd=2
            elif order == "TRIPLE":
                bd=3
            elif order == "AROMATIC": # 芳香性
                bd=10
            bonds.append((symbol1,idx1,symbol2,idx2,order,bd))
        return bonds

    def read_geo_from_out_files(self, file_path, VBdata): # 从输出文件读取信息

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        # 1、处理得到原子序数和坐标
        geo_data=[]
        in_geo_section = False

        for line in lines:
            line=line.strip()
            if line=="$geo" :
                in_geo_section = True
                continue
            elif line=="$end" and in_geo_section:
                break
            elif in_geo_section and line:
                geo_data.append(line.split())           

        atom_symbols=[]
        atomic_numbers = []
        geo_features = []
        mapping_dict=self.atom_map()

        for row in geo_data:
            symbol = str(row[0]).strip().capitalize()
            atom_symbols.append(symbol)
            atomic_numbers.append(mapping_dict(symbol))           
            geo_row=[float(row[1]),float(row[2]),float(row[3])]
            geo_features.append(geo_row)

        VBdata.sym=atom_symbols
        VBdata.atom_nums=torch.tensor(atomic_numbers)
        VBdata.nodes=torch.tensor(atomic_numbers).size(0) # 原子数
        VBdata.coor=torch.tensor(geo_features)

        # 2、处理Lowdin Weights 部分，得到VB结构
        weight_data=[]
        str_data=[]
        in_weight_section = False

        for line in lines:
            line=line.strip()
            if "Lowdin Weights" in line:
                in_weight_section = True
                print("Now in the Lowdin Weight section...\n")
                continue
            elif "Inverse Weights" in line and in_weight_section:
                print("Now end the weights part ")
                break
            elif in_weight_section and line:
                weights = line.split()
                weight_data.append(weights[1])
                for i, part in enumerate(weights):
                    if '1:' in part:
                        str_data.append(weights[i:])
                        break

        VBdata.LowdinWeights=weight_data
        VBdata.str=str_data

        # 3、处理轨道部分，得到轨道对应的原子
        orbital_data=[]
        in_orbital_section=False

        start_marker = "******  ORBITALS IN PRIMITIVE BASIS FUNCTIONS ******"

        for line in lines:
            line=line.strip("\n")
            s = line.strip()
            if start_marker in line :
                print("Now in the Orbitals part")
                in_orbital_section = True
                continue
            if not in_orbital_section:
                continue
            if s.startswith("******") and s.endswith("******"):
                print("Now end the Orbital part()")
                break                
            if not s:
                continue
            orbital_data.append(line)
        
        orbital_matched=self.match_orbital_data(orbital_data) # 匹配轨道对应的原子
        
        VBdata.atom_from_orb = self.orb_match_atom(orbital_matched).clone().detach()
        
        # VBdata.atom_from_orb: 生成[num_orbs, 3]的tensor，第一列为轨道数，第二列为对应的原子序数，第三列为对应的原子数(即几号原子)。形如[[1,6,1],[2,6,2],[3,6,3],...,]
        # print(VBdata.atom_from_orb)
        print("read_geo_from_out_files is ok, now show the VBdata.atom_nums and VBdata.coor \n",VBdata.atom_nums, VBdata.coor)

        return VBdata

    def generate_A(self,VBdata): 
        """生成邻接矩阵和边特征"""

        num_graph=len(VBdata.str) # 每个分子的结构数，即这个分子的图的数量
        nodes=VBdata.nodes
        Amat=torch.zeros(num_graph, nodes, nodes, dtype=torch.int)
        A_BO_inact=torch.zeros(nodes, nodes, dtype=torch.int)    # 所有的结构，A_BO_inact都是一样的
        A_BO_act=torch.zeros(num_graph, nodes, nodes, dtype=torch.int)
        BO_if_changed=torch.zeros(nodes, nodes, dtype=torch.int) # 用来判断该非活性键级是否修改过，避免累计
        
        # 得到一个通过RDkit形成的键级矩阵，非活性部分解决
        bonds=self.get_bond_info(VBdata.sym, VBdata.coor)

        for parts in bonds:  
            # print(parts[1]-1," ",parts[3]-1, " + ", parts[5])
            A_BO_inact[parts[1]-1][parts[3]-1]=A_BO_inact[parts[3]-1][parts[1]-1]=parts[5]
            # print("A_BO_inact: before change\n",A_BO_inact)
            for i in  range(num_graph):
                Amat[i][parts[1]-1][parts[3]-1]=Amat[i][parts[3]-1][parts[1]-1]=1
        
        # 开始遍历活性轨道对应的原子，如果经验键级矩阵的这部分原子之间成双键/多键，那么σ键就是非活性的；如果经验键级矩阵的原子之间只有单键，那么σ键就是活性的  
        for idx, parts in enumerate(VBdata.str):
            # 首先判断该分子的活性部分是否具有共轭性
            if_cov=0
            for chars in parts:
                if "-" in chars: 
                    orb_1, orb_2 = chars.split('-')
                    atom_1=VBdata.atom_from_orb[int(orb_1)-1, 2]
                    atom_2=VBdata.atom_from_orb[int(orb_2)-1, 2]
                    if A_BO_inact[atom_1-1, atom_2-1]>1:
                        if_cov=1
                        break
            if if_cov==1: break


        for idx, parts in enumerate(VBdata.str):
            for chars in parts:
                if "-" in chars: 
                    orb_1, orb_2 = chars.split('-')
                    atom_1=VBdata.atom_from_orb[int(orb_1)-1, 2]
                    atom_2=VBdata.atom_from_orb[int(orb_2)-1, 2]
                    Amat[idx, atom_1-1, atom_2-1] = 1
                    Amat[idx, atom_2-1, atom_1-1] = 1
                    A_BO_act[idx, atom_1-1, atom_2-1] += 1
                    A_BO_act[idx, atom_2-1, atom_1-1] += 1

                    if A_BO_inact[atom_1-1, atom_2-1]==1 and BO_if_changed[atom_1-1, atom_2-1]==0 and if_cov==0 : 
                        # 说明活性轨道对应的经验键级是1，那么这个键就是活性的，因此非活性键级矩阵A_BO_inact应该修改为0
                        # print("A_BO_inact[atom_1-1, atom_2-1]", atom_1-1, atom_2-1, A_BO_inact[atom_1-1, atom_2-1])
                        A_BO_inact[atom_1-1, atom_2-1]=A_BO_inact[atom_2-1, atom_1-1]=0
                        BO_if_changed[atom_1-1, atom_2-1]=1
                    elif A_BO_inact[atom_1-1, atom_2-1]>1 and BO_if_changed[atom_1-1, atom_2-1]==0 and if_cov==1 : 
                        # 说明活性轨道对应的经验键级大于1，那么这个键中，大于1的键级是活性的，σ键是非活性的，因此非活性键级矩阵A_BO_inact应该修改为1
                        A_BO_inact[atom_1-1, atom_2-1]=A_BO_inact[atom_2-1, atom_1-1]=1
                        BO_if_changed[atom_1-1, atom_2-1]=1

        # print("Amat\n",Amat[13],"\n")
        # print("A_BO_act\n",A_BO_act[13],"\n")
        # print("A_BO_inact\n",A_BO_inact,"\n")

        A_list=[[],[]]
        E=[]

        for str_idx in range(num_graph):
            for i in range(VBdata.nodes):
                for j in range(VBdata.nodes):
                    if Amat[str_idx][i][j]:
                        A_list[0].append(i+1)
                        A_list[1].append(j+1)

                        relative_coor=[VBdata.coor[j]-VBdata.coor[i]]
                        edge_row=[2 * A_BO_act[str_idx][i][j].item(), 2 * A_BO_inact[i][j].item(),relative_coor[0][0].item(),relative_coor[0][1].item(),relative_coor[0][2].item()]
                        E.append(edge_row)
                        # E:num_edge*5,这条边的活性电子数，这条边的非活性电子数，这条边的三个相对坐标（用于等变性）
        
        VBdata.A_mat=Amat
        VBdata.A_list=A_list
        VBdata.E=E
        return VBdata

    def generate_X(self, VBdata): 
        """ 生成节点特征矩阵 X: [nodes, 3*str], 第一列为每个原子的原子序数，第二列为每个原子的活性电子数，第三列为每个原子的非活性电子数 """
        X=None

        print("Now begin generate_X...\n",VBdata.atom_from_orb,"\n",VBdata.str)
        for index, parts in enumerate(VBdata.str):
            X_part=torch.zeros(VBdata.nodes,3)
            X_part[:,0]=VBdata.atom_nums
            for chars in parts:
                if "1:" in chars: 
                    continue
                elif "-" in chars:
                    orb_1, orb_2 = chars.split('-')
                    X_part[VBdata.atom_from_orb[int(orb_1)-1,2]-1,1]+=1
                    X_part[VBdata.atom_from_orb[int(orb_2)-1,2]-1,1]+=1
                else:
                    orb_3=int(chars)
                    X_part[VBdata.atom_from_orb[int(orb_3)-1,2]-1,1]+=1

            X_part[:,2] = X_part[:,0]-X_part[:,1]
            
            if X is None:
                X=X_part
            else:
                X=torch.cat([X,X_part],dim=1)

        VBdata.X=X
        print(f"VBdata.X:\n{VBdata.X[0]}")
        return VBdata

    def process_single_file(self, file_path):
        """处理单个.out文件"""
        VBdata=VBinformation()
        VBdata.molecule_id=file_path.stem
        VBdata=self.read_geo_from_out_files(file_path, VBdata)
        VBdata=self.generate_A(VBdata)
        VBdata=self.generate_X(VBdata)

        return VBdata

    def process_directory(self, directory_path):
        """处理目录中的所有.out文件"""
        directory=Path(directory_path)
        out_files = list(directory.glob("*.out")) + list(directory.glob("*.xmo"))

        for file_path in out_files:
            print(f"处理文件：{file_path.name}")
            VBdata=self.process_single_file(file_path)
            if VBdata:
                self.all_molecules.append(VBdata)
                print(f"成功处理{file_path.name}, 包含{len(VBdata.str)}个价键结构")

    def get_all_molecules(self):
        """获取所有处理好的分子数据"""
        return self.all_molecules
    
    def save_to_file(self, output_path="/pool1/home/ysxin/E3nnVB/VB/raw/vb_data_collection.pt"):
        """将处理好的数据保存到文件"""
        data_to_save = {
            'molecules': self.all_molecules,
            'num_molecules': len(self.all_molecules)
        }
        torch.save(data_to_save, output_path)
        print(f"数据已保存到 {output_path}")

if __name__ == "__main__":
    collector=VBinfo_collector()
    collector.process_directory("/pool1/home/ysxin/E3nnVB/VB/out_file")
    all_molecules=collector.get_all_molecules()
    collector.save_to_file()
