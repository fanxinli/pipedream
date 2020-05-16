import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
import torch.utils.checkpoint as cp

class BertPartitioned(torch.nn.Module):
    def __init__(self):
        super(BertPartitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()

    

    def forward(self, input0, input1, input2):
        (out18, out0) = cp.checkpoint(self.stage0,input0, input1, input2)
        print("here")
        out19 = self.stage1(out18, out0)
        (out20, out21) = self.stage2(out19, out18)
        (out23, out22) = self.stage3(out20, out21, out18)
        return (out23, out22)
