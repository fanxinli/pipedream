from .bert import BertPartitioned
from .stage0_d import Stage0
from .stage1_d import Stage1
from .stage2_d import Stage2
from .stage3_d import Stage3

def arch():
    return "bert"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out18", "out0"]),
        (Stage1(), ["out18", "out0"], ["out19", "out18"]),
        (Stage2(), ["out19", "out18"], ["out20", "out21"]),
        (Stage3(), ["out20", "out21"], ["out23"]),
        (criterion, ["out23"], ["loss"])
    ]

def full_model():
    return BertPartitioned()
