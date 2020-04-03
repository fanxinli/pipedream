from .bert import BertPartitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "bert"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out18", "out0"]),
        (Stage1(), ["out18", "out0"], ["out19"]),
        (Stage2(), ["out19", "out18"], ["out20", "out21"]),
        (Stage3(), ["out20", "out21", "out18"], ["out23", "out22"]),
        (criterion, ["out23", "out22"], ["loss"])
    ]

def full_model():
    return BertPartitioned()
