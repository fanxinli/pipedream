import torch
from modeling import BertLayerNorm
from modeling import BertSelfAttention
from modeling import BertEmbeddings
from modeling import LinearActivation

import torch.utils.checkpoint as cp



def _bn_function_factory(linear_activation, linear, dropout):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

def _test_function_factory(linear, dropout, norm):
    def test_function(*inputs):

        output= norm(inputs[0]+dropout(linear(inputs[1])))

        return output
    return test_function

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer6 = BertEmbeddings(30528, 1024, 512, 2, 0.1)
        self.layer7 = BertSelfAttention(1024, 16, 0.1)

        self.layer8 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer9 = torch.nn.Dropout(p=0.1)
        self.layer11 = BertLayerNorm(1024)


        # self.layer12 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer13 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer14 = torch.nn.Dropout(p=0.1)

        self.seq1 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))

        self.layer16 = BertLayerNorm(1024)
        self.layer17 = BertSelfAttention(1024, 16, 0.1)
        self.layer18 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer19 = torch.nn.Dropout(p=0.1)
        self.layer21 = BertLayerNorm(1024)
        # self.layer22 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer23 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer24 = torch.nn.Dropout(p=0.1)

        self.seq2 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))


        self.layer26 = BertLayerNorm(1024)
        self.layer27 = BertSelfAttention(1024, 16, 0.1)
        self.layer28 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer29 = torch.nn.Dropout(p=0.1)
        self.layer31 = BertLayerNorm(1024)
        # self.layer32 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer33 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer34 = torch.nn.Dropout(p=0.1)
        self.seq3 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))

        self.layer36 = BertLayerNorm(1024)
        self.layer37 = BertSelfAttention(1024, 16, 0.1)
        self.layer38 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer39 = torch.nn.Dropout(p=0.1)
        self.layer41 = BertLayerNorm(1024)
        # self.layer42 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer43 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer44 = torch.nn.Dropout(p=0.1)
        self.seq4 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))

        self.layer46 = BertLayerNorm(1024)
        self.layer47 = BertSelfAttention(1024, 16, 0.1)
        self.layer48 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer49 = torch.nn.Dropout(p=0.1)
        self.layer51 = BertLayerNorm(1024)

        # self.layer52 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer53 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer54 = torch.nn.Dropout(p=0.1)
        self.seq5 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))

        self.layer56 = BertLayerNorm(1024)
        self.layer57 = BertSelfAttention(1024, 16, 0.1)
        self.layer58 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer59 = torch.nn.Dropout(p=0.1)
        self.layer61 = BertLayerNorm(1024)

        # self.layer62 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        # self.layer63 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        # self.layer64 = torch.nn.Dropout(p=0.1)
        self.seq6 =  torch.nn.Sequential(LinearActivation(in_features=1024, out_features=4096, bias=True), torch.nn.Linear(in_features=4096, out_features=1024, bias=True), torch.nn.Dropout(p=0.1))

    

    def forward(self, input0, input1, input2):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out6 = self.layer6(out0, out1)
        out7 = self.layer7(out6, out2)

        test_func1 = _test_function_factory(self.layer8, self.layer9, self.layer11)
        # out8 = self.layer8(out7)
        # out9 = self.layer9(out8)
        # out9 = out9 + out6
        out11 = cp.checkpoint(test_func1, out6, out7)   #self.layer11(out9)


        # out12 = self.layer12(out11)
        # out13 = self.layer13(out12)
        out14 = cp.checkpoint_sequential(self.seq1, 3, out11)#self.layer14(out13)
        out14 = out14 + out11
        out16 = self.layer16(out14)
        out17 = self.layer17(out16, out2)

        test_func2 = _test_function_factory(self.layer18, self.layer19, self.layer21)

        # out18 = self.layer18(out17)
        # out19 = self.layer19(out18)
        # out19 = out19 + out16
        out21 = cp.checkpoint(test_func2, out16, out17)#self.layer21(out19)

        # out22 = self.layer22(out21)
        # out23 = self.layer23(out22)
        out24 = cp.checkpoint_sequential(self.seq2, 3, out21) #self.layer24(out23)
        out24 = out24 + out21
        out26 = self.layer26(out24)
        out27 = self.layer27(out26, out2)

        test_func3 = _test_function_factory(self.layer28, self.layer29, self.layer31)

        # out28 = self.layer28(out27)
        # out29 = self.layer29(out28)
        # out29 = out29 + out26
        out31 = cp.checkpoint(test_func3, out26, out27)#self.layer31(out29)
        # out32 = self.layer32(out31)
        # out33 = self.layer33(out32)
        out34 = cp.checkpoint_sequential(self.seq3, 3, out31) #self.layer34(out33)
        out34 = out34 + out31
        out36 = self.layer36(out34)
        out37 = self.layer37(out36, out2)

        test_func4 = _test_function_factory(self.layer38, self.layer39, self.layer41)
        
        # out38 = self.layer38(out37)
        # out39 = self.layer39(out38)
        # out39 = out39 + out36
        out41 = cp.checkpoint(test_func4, out36, out37)#self.layer41(out39)
        # out42 = self.layer42(out41)
        # out43 = self.layer43(out42)
        out44 = cp.checkpoint_sequential(self.seq4, 3, out41)#self.layer44(out43)
        out44 = out44 + out41
        out46 = self.layer46(out44)
        out47 = self.layer47(out46, out2)

        test_func5 = _test_function_factory(self.layer48, self.layer49, self.layer51)

        # out48 = self.layer48(out47)
        # out49 = self.layer49(out48)
        # out49 = out49 + out46
        out51 = cp.checkpoint(test_func5, out46, out47)#self.layer51(out49)

        # out52 = self.layer52(out51)
        # out53 = self.layer53(out52)
        out54 = cp.checkpoint_sequential(self.seq5, 3, out51)#self.layer54(out53)
        out54 = out54 + out51
        out56 = self.layer56(out54)
        out57 = self.layer57(out56, out2)

        # out58 = self.layer58(out57)
        # out59 = self.layer59(out58)
        # out59 = out59 + out56
        test_func6 = _test_function_factory(self.layer58, self.layer59, self.layer61)

        out61 = cp.checkpoint(test_func6, out56, out57)#self.layer61(out59)
        # out62 = self.layer62(out61)
        # out63 = self.layer63(out62)
        out64 = cp.checkpoint_sequential(self.seq6, 3, out61)#self.layer64(out63)
        out64 = out64 + out61
        return (out2, out64)
