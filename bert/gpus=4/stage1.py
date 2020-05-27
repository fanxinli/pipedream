import torch
from modeling import BertLayerNorm
from modeling import BertSelfAttention
from modeling import LinearActivation


import torch.utils.checkpoint as cp

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = BertLayerNorm(1024)
        self.layer3 = BertSelfAttention(1024, 16, 0.1)
        self.layer4 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer5 = torch.nn.Dropout(p=0.1)
        self.layer7 = BertLayerNorm(1024)
        self.layer8 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer9 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer10 = torch.nn.Dropout(p=0.1)
        self.layer12 = BertLayerNorm(1024)
        self.layer13 = BertSelfAttention(1024, 16, 0.1)
        self.layer14 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer15 = torch.nn.Dropout(p=0.1)
        self.layer17 = BertLayerNorm(1024)
        self.layer18 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer19 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer20 = torch.nn.Dropout(p=0.1)
        self.layer22 = BertLayerNorm(1024)
        self.layer23 = BertSelfAttention(1024, 16, 0.1)
        self.layer24 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer25 = torch.nn.Dropout(p=0.1)
        self.layer27 = BertLayerNorm(1024)
        self.layer28 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer29 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer30 = torch.nn.Dropout(p=0.1)
        self.layer32 = BertLayerNorm(1024)
        self.layer33 = BertSelfAttention(1024, 16, 0.1)
        self.layer34 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer35 = torch.nn.Dropout(p=0.1)
        self.layer37 = BertLayerNorm(1024)
        self.layer38 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer39 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer40 = torch.nn.Dropout(p=0.1)
        self.layer42 = BertLayerNorm(1024)
        self.layer43 = BertSelfAttention(1024, 16, 0.1)
        self.layer44 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer45 = torch.nn.Dropout(p=0.1)
        self.layer47 = BertLayerNorm(1024)
        self.layer48 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer49 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer50 = torch.nn.Dropout(p=0.1)
        self.layer52 = BertLayerNorm(1024)
        self.layer53 = BertSelfAttention(1024, 16, 0.1)
        self.layer54 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer55 = torch.nn.Dropout(p=0.1)
        self.layer57 = BertLayerNorm(1024)
        self.layer58 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer59 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer60 = torch.nn.Dropout(p=0.1)
        self.layer62 = BertLayerNorm(1024)
        self.layer63 = BertSelfAttention(1024, 16, 0.1)
        self.layer64 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer65 = torch.nn.Dropout(p=0.1)
        self.layer67 = BertLayerNorm(1024)
        self.layer68 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer69 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer70 = torch.nn.Dropout(p=0.1)

        self.apply(self.init_bert_weights)


    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, input7, input0):
        return cp.checkpoint(self.forward_, input7, input0)


    def forward_(self, input7, input0):
        out0 = input0.clone()
        out1 = input7.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2, out1)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out5 = out5 + out2
        out7 = self.layer7(out5)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out10 = out10 + out7
        out12 = self.layer12(out10)
        out13 = self.layer13(out12, out1)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out15 = out15 + out12
        out17 = self.layer17(out15)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out20 = out20 + out17
        out22 = self.layer22(out20)
        out23 = self.layer23(out22, out1)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out25 = out25 + out22
        out27 = self.layer27(out25)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out30 = out30 + out27
        out32 = self.layer32(out30)
        out33 = self.layer33(out32, out1)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out35 = out35 + out32
        out37 = self.layer37(out35)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out40 = out40 + out37
        out42 = self.layer42(out40)
        out43 = self.layer43(out42, out1)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out45 = out45 + out42
        out47 = self.layer47(out45)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out50 = out50 + out47
        out52 = self.layer52(out50)
        out53 = self.layer53(out52, out1)
        out54 = self.layer54(out53)
        out55 = self.layer55(out54)
        out55 = out55 + out52
        out57 = self.layer57(out55)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out60 = out60 + out57
        out62 = self.layer62(out60)
        out63 = self.layer63(out62, out1)
        out64 = self.layer64(out63)
        out65 = self.layer65(out64)
        out65 = out65 + out62
        out67 = self.layer67(out65)
        out68 = self.layer68(out67)
        out69 = self.layer69(out68)
        out70 = self.layer70(out69)
        out70 = out70 + out67
        return (out70, out1)