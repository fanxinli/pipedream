import torch
from modeling import BertLayerNorm
from modeling import BertSelfAttention
from modeling import BertPooler
from modeling import BertAdd
from modeling import LinearActivation

class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer3 = torch.nn.Dropout(p=0.1)
        self.layer5 = BertLayerNorm(1024)
        self.layer6 = BertSelfAttention(1024, 16, 0.1)
        self.layer7 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer8 = torch.nn.Dropout(p=0.1)
        self.layer10 = BertLayerNorm(1024)
        self.layer11 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer12 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer13 = torch.nn.Dropout(p=0.1)
        self.layer15 = BertLayerNorm(1024)
        self.layer16 = BertSelfAttention(1024, 16, 0.1)
        self.layer17 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer18 = torch.nn.Dropout(p=0.1)
        self.layer20 = BertLayerNorm(1024)
        self.layer21 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer22 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer23 = torch.nn.Dropout(p=0.1)
        self.layer25 = BertLayerNorm(1024)
        self.layer26 = BertSelfAttention(1024, 16, 0.1)
        self.layer27 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer28 = torch.nn.Dropout(p=0.1)
        self.layer30 = BertLayerNorm(1024)
        self.layer31 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer32 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer33 = torch.nn.Dropout(p=0.1)
        self.layer35 = BertLayerNorm(1024)
        self.layer36 = BertSelfAttention(1024, 16, 0.1)
        self.layer37 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer38 = torch.nn.Dropout(p=0.1)
        self.layer40 = BertLayerNorm(1024)
        self.layer41 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer42 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer43 = torch.nn.Dropout(p=0.1)
        self.layer45 = BertLayerNorm(1024)
        self.layer46 = LinearActivation(in_features=1024, out_features=1024, bias=True)
        self.layer47 = BertLayerNorm(1024)
        self.layer48 = torch.nn.Linear(in_features=1024, out_features=30528, bias=False)
        self.layer49 = BertAdd(30528)
        self.layer50 = BertPooler(1024)
        self.layer51 = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    

    def forward(self, input1, input0, input5):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input5.clone()
        out3 = self.layer3(out0)
        out3 = out3 + out1
        out5 = self.layer5(out3)
        out6 = self.layer6(out5, out2)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out8 = out8 + out5
        out10 = self.layer10(out8)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out13 = out13 + out10
        out15 = self.layer15(out13)
        out16 = self.layer16(out15, out2)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out18 = out18 + out15
        out20 = self.layer20(out18)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out23 = out23 + out20
        out25 = self.layer25(out23)
        out26 = self.layer26(out25, out2)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out28 = out28 + out25
        out30 = self.layer30(out28)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out33 = out33 + out30
        out35 = self.layer35(out33)
        out36 = self.layer36(out35, out2)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out38 = out38 + out35
        out40 = self.layer40(out38)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out43 = out43 + out40
        out45 = self.layer45(out43)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out45)
        out51 = self.layer51(out50)
        return (out49, out51, out2)
