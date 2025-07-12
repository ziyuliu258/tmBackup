import torch
import pandas as pd

# -------------------- 自定义包装类 --------------------

class CustomBool:
    def __init__(self, value):
        self.value = bool(value)

    def __repr__(self):
        return f'{{bool}} {self.value}'

class CustomInt:
    def __init__(self, value):
        self.value = int(value)

    def __repr__(self):
        return f'{{int}} {self.value}'

class CustomStr:
    def __init__(self, value):
        self.value = str(value)

    def __repr__(self):
        return f'{{str}} {self.value}'

# 自定义 list 和 dict 子类
class CustomList(list):
    def __repr__(self):
        return f'{{list: {len(self)}}} {super().__repr__()}'

class CustomDict(dict):
    def __repr__(self):
        return f'{{dict: {len(self)}}} {super().__repr__()}'

# 自定义 Tensor 的 __repr__ (Torch)
original_tensor_repr = torch.Tensor.__repr__
def custom_tensor_repr(self):
    return f'{{Tensor: {tuple(self.shape)}}} {original_tensor_repr(self)}'
torch.Tensor.__repr__ = custom_tensor_repr

# 自定义 DataFrame 的 __repr__ (Pandas)
original_dataframe_repr = pd.DataFrame.__repr__
def custom_dataframe_repr(self):
    return f'{{DataFrame: {self.shape}}} {original_dataframe_repr(self)}'
pd.DataFrame.__repr__ = custom_dataframe_repr

# 自定义 DataLoader 的类
class DataLoader:
    def __init__(self, data_size):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __repr__(self):
        return f'{{DataLoader: {len(self)}}} DataLoader object'

# -------------------- __main__ 函数 --------------------
def main():
    # 使用自定义类型代替原生类型
    my_list = CustomList([1, 2, 3, 4, 5, 6])
    my_dict = CustomDict({'a': 1, 'b': 2, 'c': 3})
    my_bool = CustomBool(True)
    my_int = CustomInt(42)
    my_str = CustomStr("hello")

    # 测试 Tensor
    my_tensor = torch.randn(100, 512)

    # 测试 DataFrame
    my_dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # 测试 DataLoader
    my_dataloader = DataLoader(220)

    # 输出内容
    print(my_list)        # {list: 6} [1, 2, 3, 4, 5, 6]
    print(my_dict)        # {dict: 3} {'a': 1, 'b': 2, 'c': 3}
    print(my_bool)        # {bool} True
    print(my_int)         # {int} 42
    print(my_str)         # {str} 'hello'
    print(my_tensor)      # {Tensor: (100, 512)} tensor([...])
    print(my_dataframe)   # {DataFrame: (3, 3)}    A  B  C
    print(my_dataloader)  # {DataLoader: 220} DataLoader object

# 如果是直接运行文件，则调用 main 函数
if __name__ == "__main__":
    main()
