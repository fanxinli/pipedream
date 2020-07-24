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

class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.last = 119
        self.start = 121
        self.end = 179 ## layer id
        self.b = 2



        self.declares = '''
self.layer6 = BertEmbeddings(30528, 1024, 512, 2, 0.1)
self.layer7 = BertSelfAttention(1024, 16, 0.1)
self.layer8 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer9 = torch.nn.Dropout(p=0.1)
self.layer11 = BertLayerNorm(1024)
self.layer12 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer13 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer14 = torch.nn.Dropout(p=0.1)
self.layer16 = BertLayerNorm(1024)
self.layer17 = BertSelfAttention(1024, 16, 0.1)
self.layer18 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer19 = torch.nn.Dropout(p=0.1)
self.layer21 = BertLayerNorm(1024)
self.layer22 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer23 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer24 = torch.nn.Dropout(p=0.1)
self.layer26 = BertLayerNorm(1024)
self.layer27 = BertSelfAttention(1024, 16, 0.1)
self.layer28 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer29 = torch.nn.Dropout(p=0.1)
self.layer31 = BertLayerNorm(1024)
self.layer32 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer33 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer34 = torch.nn.Dropout(p=0.1)
self.layer36 = BertLayerNorm(1024)
self.layer37 = BertSelfAttention(1024, 16, 0.1)
self.layer38 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer39 = torch.nn.Dropout(p=0.1)
self.layer41 = BertLayerNorm(1024)
self.layer42 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer43 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer44 = torch.nn.Dropout(p=0.1)
self.layer46 = BertLayerNorm(1024)
self.layer47 = BertSelfAttention(1024, 16, 0.1)
self.layer48 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer49 = torch.nn.Dropout(p=0.1)
self.layer51 = BertLayerNorm(1024)
self.layer52 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer53 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer54 = torch.nn.Dropout(p=0.1)
self.layer56 = BertLayerNorm(1024)
self.layer57 = BertSelfAttention(1024, 16, 0.1)
self.layer58 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer59 = torch.nn.Dropout(p=0.1)
self.layer61 = BertLayerNorm(1024)
self.layer62 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer63 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer64 = torch.nn.Dropout(p=0.1)
self.layer66 = BertLayerNorm(1024)
self.layer67 = BertSelfAttention(1024, 16, 0.1)
self.layer68 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer69 = torch.nn.Dropout(p=0.1)
self.layer71 = BertLayerNorm(1024)
self.layer72 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer73 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer74 = torch.nn.Dropout(p=0.1)
self.layer76 = BertLayerNorm(1024)
self.layer77 = BertSelfAttention(1024, 16, 0.1)
self.layer78 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer79 = torch.nn.Dropout(p=0.1)
self.layer81 = BertLayerNorm(1024)
self.layer82 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer83 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer84 = torch.nn.Dropout(p=0.1)
self.layer86 = BertLayerNorm(1024)
self.layer87 = BertSelfAttention(1024, 16, 0.1)
self.layer88 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer89 = torch.nn.Dropout(p=0.1)
self.layer91 = BertLayerNorm(1024)
self.layer92 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer93 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer94 = torch.nn.Dropout(p=0.1)
self.layer96 = BertLayerNorm(1024)
self.layer97 = BertSelfAttention(1024, 16, 0.1)
self.layer98 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer99 = torch.nn.Dropout(p=0.1)
self.layer101 = BertLayerNorm(1024)
self.layer102 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer103 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer104 = torch.nn.Dropout(p=0.1)
self.layer106 = BertLayerNorm(1024)
self.layer107 = BertSelfAttention(1024, 16, 0.1)
self.layer108 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer109 = torch.nn.Dropout(p=0.1)
self.layer111 = BertLayerNorm(1024)
self.layer112 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer113 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer114 = torch.nn.Dropout(p=0.1)
self.layer116 = BertLayerNorm(1024)
self.layer117 = BertSelfAttention(1024, 16, 0.1)
self.layer118 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer119 = torch.nn.Dropout(p=0.1)
self.layer121 = BertLayerNorm(1024)
self.layer122 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer123 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer124 = torch.nn.Dropout(p=0.1)
self.layer126 = BertLayerNorm(1024)
self.layer127 = BertSelfAttention(1024, 16, 0.1)
self.layer128 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer129 = torch.nn.Dropout(p=0.1)
self.layer131 = BertLayerNorm(1024)
self.layer132 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer133 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer134 = torch.nn.Dropout(p=0.1)
self.layer136 = BertLayerNorm(1024)
self.layer137 = BertSelfAttention(1024, 16, 0.1)
self.layer138 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer139 = torch.nn.Dropout(p=0.1)
self.layer141 = BertLayerNorm(1024)
self.layer142 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer143 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer144 = torch.nn.Dropout(p=0.1)
self.layer146 = BertLayerNorm(1024)
self.layer147 = BertSelfAttention(1024, 16, 0.1)
self.layer148 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer149 = torch.nn.Dropout(p=0.1)
self.layer151 = BertLayerNorm(1024)
self.layer152 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer153 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer154 = torch.nn.Dropout(p=0.1)
self.layer156 = BertLayerNorm(1024)
self.layer157 = BertSelfAttention(1024, 16, 0.1)
self.layer158 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer159 = torch.nn.Dropout(p=0.1)
self.layer161 = BertLayerNorm(1024)
self.layer162 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer163 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer164 = torch.nn.Dropout(p=0.1)
self.layer166 = BertLayerNorm(1024)
self.layer167 = BertSelfAttention(1024, 16, 0.1)
self.layer168 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer169 = torch.nn.Dropout(p=0.1)
self.layer171 = BertLayerNorm(1024)
self.layer172 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer173 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer174 = torch.nn.Dropout(p=0.1)
self.layer176 = BertLayerNorm(1024)
self.layer177 = BertSelfAttention(1024, 16, 0.1)
self.layer178 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer179 = torch.nn.Dropout(p=0.1)
self.layer181 = BertLayerNorm(1024)
self.layer182 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer183 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer184 = torch.nn.Dropout(p=0.1)
self.layer186 = BertLayerNorm(1024)
self.layer187 = BertSelfAttention(1024, 16, 0.1)
self.layer188 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer189 = torch.nn.Dropout(p=0.1)
self.layer191 = BertLayerNorm(1024)
self.layer192 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer193 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer194 = torch.nn.Dropout(p=0.1)
self.layer196 = BertLayerNorm(1024)
self.layer197 = BertSelfAttention(1024, 16, 0.1)
self.layer198 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer199 = torch.nn.Dropout(p=0.1)
self.layer201 = BertLayerNorm(1024)
self.layer202 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer203 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer204 = torch.nn.Dropout(p=0.1)
self.layer206 = BertLayerNorm(1024)
self.layer207 = BertSelfAttention(1024, 16, 0.1)
self.layer208 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer209 = torch.nn.Dropout(p=0.1)
self.layer211 = BertLayerNorm(1024)
self.layer212 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer213 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer214 = torch.nn.Dropout(p=0.1)
self.layer216 = BertLayerNorm(1024)
self.layer217 = BertSelfAttention(1024, 16, 0.1)
self.layer218 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer219 = torch.nn.Dropout(p=0.1)
self.layer221 = BertLayerNorm(1024)
self.layer222 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer223 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer224 = torch.nn.Dropout(p=0.1)
self.layer226 = BertLayerNorm(1024)
self.layer227 = BertSelfAttention(1024, 16, 0.1)
self.layer228 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer229 = torch.nn.Dropout(p=0.1)
self.layer231 = BertLayerNorm(1024)
self.layer232 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer233 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer234 = torch.nn.Dropout(p=0.1)
self.layer236 = BertLayerNorm(1024)
self.layer237 = BertSelfAttention(1024, 16, 0.1)
self.layer238 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer239 = torch.nn.Dropout(p=0.1)
self.layer241 = BertLayerNorm(1024)
self.layer242 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer243 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer244 = torch.nn.Dropout(p=0.1)
self.layer246 = BertLayerNorm(1024)
self.layer247 = LinearActivation(in_features=1024, out_features=1024, bias=True)
self.layer248 = BertLayerNorm(1024)
self.layer249 = torch.nn.Linear(in_features=1024, out_features=30528, bias=False)
        '''

        self.generate_declares()
        exec(self.declare)

        self.apply(self.init_bert_weights)

        self.stmts = '''
out6 = self.layer6(out0, out1)
out7 = self.layer7(out6, out2)
out8 = self.layer8(out7)
out9 = self.layer9(out8)
out9 = out9 + out6
out11 = self.layer11(out9)
out12 = self.layer12(out11)
out13 = self.layer13(out12)
out14 = self.layer14(out13)
out14 = out14 + out11
out16 = self.layer16(out14)
out17 = self.layer17(out16, out2)
out18 = self.layer18(out17)
out19 = self.layer19(out18)
out19 = out19 + out16
out21 = self.layer21(out19)
out22 = self.layer22(out21)
out23 = self.layer23(out22)
out24 = self.layer24(out23)
out24 = out24 + out21
out26 = self.layer26(out24)
out27 = self.layer27(out26, out2)
out28 = self.layer28(out27)
out29 = self.layer29(out28)
out29 = out29 + out26
out31 = self.layer31(out29)
out32 = self.layer32(out31)
out33 = self.layer33(out32)
out34 = self.layer34(out33)
out34 = out34 + out31
out36 = self.layer36(out34)
out37 = self.layer37(out36, out2)
out38 = self.layer38(out37)
out39 = self.layer39(out38)
out39 = out39 + out36
out41 = self.layer41(out39)
out42 = self.layer42(out41)
out43 = self.layer43(out42)
out44 = self.layer44(out43)
out44 = out44 + out41
out46 = self.layer46(out44)
out47 = self.layer47(out46, out2)
out48 = self.layer48(out47)
out49 = self.layer49(out48)
out49 = out49 + out46
out51 = self.layer51(out49)
out52 = self.layer52(out51)
out53 = self.layer53(out52)
out54 = self.layer54(out53)
out54 = out54 + out51
out56 = self.layer56(out54)
out57 = self.layer57(out56, out2)
out58 = self.layer58(out57)
out59 = self.layer59(out58)
out59 = out59 + out56
out61 = self.layer61(out59)
out62 = self.layer62(out61)
out63 = self.layer63(out62)
out64 = self.layer64(out63)
out64 = out64 + out61
out66 = self.layer66(out64)
out67 = self.layer67(out66, out2)
out68 = self.layer68(out67)
out69 = self.layer69(out68)
out69 = out69 + out66
out71 = self.layer71(out69)
out72 = self.layer72(out71)
out73 = self.layer73(out72)
out74 = self.layer74(out73)
out74 = out74 + out71
out76 = self.layer76(out74)
out77 = self.layer77(out76, out2)
out78 = self.layer78(out77)
out79 = self.layer79(out78)
out79 = out79 + out76
out81 = self.layer81(out79)
out82 = self.layer82(out81)
out83 = self.layer83(out82)
out84 = self.layer84(out83)
out84 = out84 + out81
out86 = self.layer86(out84)
out87 = self.layer87(out86, out2)
out88 = self.layer88(out87)
out89 = self.layer89(out88)
out89 = out89 + out86
out91 = self.layer91(out89)
out92 = self.layer92(out91)
out93 = self.layer93(out92)
out94 = self.layer94(out93)
out94 = out94 + out91
out96 = self.layer96(out94)
out97 = self.layer97(out96, out2)
out98 = self.layer98(out97)
out99 = self.layer99(out98)
out99 = out99 + out96
out101 = self.layer101(out99)
out102 = self.layer102(out101)
out103 = self.layer103(out102)
out104 = self.layer104(out103)
out104 = out104 + out101
out106 = self.layer106(out104)
out107 = self.layer107(out106, out2)
out108 = self.layer108(out107)
out109 = self.layer109(out108)
out109 = out109 + out106
out111 = self.layer111(out109)
out112 = self.layer112(out111)
out113 = self.layer113(out112)
out114 = self.layer114(out113)
out114 = out114 + out111
out116 = self.layer116(out114)
out117 = self.layer117(out116, out2)
out118 = self.layer118(out117)
out119 = self.layer119(out118)
out119 = out119 + out116
out121 = self.layer121(out119)
out122 = self.layer122(out121)
out123 = self.layer123(out122)
out124 = self.layer124(out123)
out124 = out124 + out121
out126 = self.layer126(out124)
out127 = self.layer127(out126, out2)
out128 = self.layer128(out127)
out129 = self.layer129(out128)
out129 = out129 + out126
out131 = self.layer131(out129)
out132 = self.layer132(out131)
out133 = self.layer133(out132)
out134 = self.layer134(out133)
out134 = out134 + out131
out136 = self.layer136(out134)
out137 = self.layer137(out136, out2)
out138 = self.layer138(out137)
out139 = self.layer139(out138)
out139 = out139 + out136
out141 = self.layer141(out139)
out142 = self.layer142(out141)
out143 = self.layer143(out142)
out144 = self.layer144(out143)
out144 = out144 + out141
out146 = self.layer146(out144)
out147 = self.layer147(out146, out2)
out148 = self.layer148(out147)
out149 = self.layer149(out148)
out149 = out149 + out146
out151 = self.layer151(out149)
out152 = self.layer152(out151)
out153 = self.layer153(out152)
out154 = self.layer154(out153)
out154 = out154 + out151
out156 = self.layer156(out154)
out157 = self.layer157(out156, out2)
out158 = self.layer158(out157)
out159 = self.layer159(out158)
out159 = out159 + out156
out161 = self.layer161(out159)
out162 = self.layer162(out161)
out163 = self.layer163(out162)
out164 = self.layer164(out163)
out164 = out164 + out161
out166 = self.layer166(out164)
out167 = self.layer167(out166, out2)
out168 = self.layer168(out167)
out169 = self.layer169(out168)
out169 = out169 + out166
out171 = self.layer171(out169)
out172 = self.layer172(out171)
out173 = self.layer173(out172)
out174 = self.layer174(out173)
out174 = out174 + out171
out176 = self.layer176(out174)
out177 = self.layer177(out176, out2)
out178 = self.layer178(out177)
out179 = self.layer179(out178)
out179 = out179 + out176
out181 = self.layer181(out179)
out182 = self.layer182(out181)
out183 = self.layer183(out182)
out184 = self.layer184(out183)
out184 = out184 + out181
out186 = self.layer186(out184)
out187 = self.layer187(out186, out2)
out188 = self.layer188(out187)
out189 = self.layer189(out188)
out189 = out189 + out186
out191 = self.layer191(out189)
out192 = self.layer192(out191)
out193 = self.layer193(out192)
out194 = self.layer194(out193)
out194 = out194 + out191
out196 = self.layer196(out194)
out197 = self.layer197(out196, out2)
out198 = self.layer198(out197)
out199 = self.layer199(out198)
out199 = out199 + out196
out201 = self.layer201(out199)
out202 = self.layer202(out201)
out203 = self.layer203(out202)
out204 = self.layer204(out203)
out204 = out204 + out201
out206 = self.layer206(out204)
out207 = self.layer207(out206, out2)
out208 = self.layer208(out207)
out209 = self.layer209(out208)
out209 = out209 + out206
out211 = self.layer211(out209)
out212 = self.layer212(out211)
out213 = self.layer213(out212)
out214 = self.layer214(out213)
out214 = out214 + out211
out216 = self.layer216(out214)
out217 = self.layer217(out216, out2)
out218 = self.layer218(out217)
out219 = self.layer219(out218)
out219 = out219 + out216
out221 = self.layer221(out219)
out222 = self.layer222(out221)
out223 = self.layer223(out222)
out224 = self.layer224(out223)
out224 = out224 + out221
out226 = self.layer226(out224)
out227 = self.layer227(out226, out2)
out228 = self.layer228(out227)
out229 = self.layer229(out228)
out229 = out229 + out226
out231 = self.layer231(out229)
out232 = self.layer232(out231)
out233 = self.layer233(out232)
out234 = self.layer234(out233)
out234 = out234 + out231
out236 = self.layer236(out234)
out237 = self.layer237(out236, out2)
out238 = self.layer238(out237)
out239 = self.layer239(out238)
out239 = out239 + out236
out241 = self.layer241(out239)
out242 = self.layer242(out241)
out243 = self.layer243(out242)
out244 = self.layer244(out243)
out244 = out244 + out241
out246 = self.layer246(out244)
out247 = self.layer247(out246)
out248 = self.layer248(out247)
out249 = self.layer249(out248)
out250 = self.layer250(out249)
'''

        self.generate_stmt()
        self.in_cuda = False

        self.generate_cp()




    def stage_to_cuda(self):
        if (not self.in_cuda):
            for i in range(self.start, self.end+1):
                if hasattr(self, "layer"+str(i)):
                    getattr(self, "layer"+str(i)).cuda()
                    print("layer"+str(i)+" to cuda")
            self.in_cuda = True

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

    def generate_declares(self):
        start = self.start
        end = self.end

        l = self.declares.split('\n')
        declare = []
        index = start

        record = False
        finishing = False

        for line in l:

            if finishing:
                break

            if "layer"+str(start) in line:
                record = True
            if (record):
                declare.append(line)

            if "layer"+str(end) in line:
                finishing = True
        self.declare = '\n'.join(declare)


    def generate_stmt(self):

        start = self.start
        end = self.end

        l = self.stmts.split('\n')
        stmt = []
        index = start
        record = False
        finishing = False

        for line in l:

            if finishing:
                if "+" in line:
                    stmt.append(line)
                break

            if "layer"+str(start) in line:
                record = True
            if (record):
                stmt.append(line)

            if "layer"+str(end) in line:
                finishing = True

        stmt.append("self.out = out"+str(end))
        stmt[0] = stmt[0].replace("out"+str(self.last), "out0")

        self.stmt = '\n'.join(stmt)

    def generate_cp(self):

        cp_ = []
        no_cp_ = []


        b = self.b

        no_cp_ = self.stmt.split('\n')[:-b]
        cp_ = self.stmt.split('\n')[-b:]

        r = []
        if "+" in no_cp_[-1]:
            r.append(no_cp_[-1].split('=')[0])
            r.append('out2')
        else:
            count = 1
            while True:
                if "+" in no_cp_[-count]:
                    break
                else:
                    count = count+1

            print(count)
            if (count == 1):
                r.append(no_cp_[-1].split('=')[0])
                r.append("out2")
            elif (count == 2):
                r.append(no_cp_[-1].split('=')[0])
                r.append("out2")
                #r.append(no_cp_[-2].split('=')[0])
            elif (count == 3):
                r.append(no_cp_[-1].split('=')[0])
                r.append("out2")
                r.append(no_cp_[-2].split('=')[0])
            elif (count == 4):
                r.append(no_cp_[-1].split('=')[0])
                r.append("out2")
                r.append(no_cp_[-3].split('=')[0])
            elif (count == 5):
                r.append(no_cp_[-1].split('=')[0])
                r.append("out2")
                r.append(no_cp_[-4].split('=')[0])
            
     
        no_cp_.append("self.out = cp.checkpoint(self.cp_forward, {})".format(','.join(r)))

        count_ = 0
        args_ = []
        for rt in r:
            args_.append(rt+"= args[{}]".format(count_))
            count_=count_+1
        
        self.cp= '\n'.join(args_+cp_)
        self.no_cp = '\n'.join(no_cp_)
    def forward(self, input0, input1):
        return self.forward_(input0, input1)
        #return cp.checkpoint(self.forward_, input0, input1)


    def forward_(self, input0, input1):
        out0 = input0.clone()
        out2 = input1.clone()
        exec(self.no_cp)
        return (self.out, out2)

    def cp_forward(self, *args):
        exec(self.cp)
        return self.out