import torch
from modeling import BertEmbeddings
from modeling import BertSelfAttention
from modeling import BertLayerNorm
from modeling import LinearActivation
from modeling import BertPooler
from modeling import BertAdd

def model(criterion):
    return [
        (Layer2(), ["out0", "out1"], ["out3"]),
        (Layer4(), ["out3", "out27"], ["out28"]),
        (Layer5(), ["out28"], ["out29"]),
        (Layer6(), ["out29"], ["out30"]),
        (Layer7(), ["out3", "out30"], ["out31"]),
        (Layer8(), ["out31"], ["out33"]),
        (Layer9(), ["out33"], ["out34"]),
        (Layer10(), ["out34"], ["out35"]),
        (Layer11(), ["out35"], ["out36"]),
        (Layer12(), ["out33", "out36"], ["out37"]),
        (Layer13(), ["out37"], ["out39"]),
        (Layer14(), ["out39", "out27"], ["out40"]),
        (Layer15(), ["out40"], ["out41"]),
        (Layer16(), ["out41"], ["out42"]),
        (Layer17(), ["out39", "out42"], ["out43"]),
        (Layer18(), ["out43"], ["out45"]),
        (Layer19(), ["out45"], ["out46"]),
        (Layer20(), ["out46"], ["out47"]),
        (Layer21(), ["out47"], ["out48"]),
        (Layer22(), ["out45", "out48"], ["out49"]),
        (Layer23(), ["out49"], ["out51"]),
        (Layer24(), ["out51", "out27"], ["out52"]),
        (Layer25(), ["out52"], ["out53"]),
        (Layer26(), ["out53"], ["out54"]),
        (Layer27(), ["out51", "out54"], ["out55"]),
        (Layer28(), ["out55"], ["out57"]),
        (Layer29(), ["out57"], ["out58"]),
        (Layer30(), ["out58"], ["out59"]),
        (Layer31(), ["out59"], ["out60"]),
        (Layer32(), ["out57", "out60"], ["out61"]),
        (Layer33(), ["out61"], ["out63"]),
        (Layer34(), ["out63", "out27"], ["out64"]),
        (Layer35(), ["out64"], ["out65"]),
        (Layer36(), ["out65"], ["out66"]),
        (Layer37(), ["out63", "out66"], ["out67"]),
        (Layer38(), ["out67"], ["out69"]),
        (Layer39(), ["out69"], ["out70"]),
        (Layer40(), ["out70"], ["out71"]),
        (Layer41(), ["out71"], ["out72"]),
        (Layer42(), ["out69", "out72"], ["out73"]),
        (Layer43(), ["out73"], ["out75"]),
        (Layer44(), ["out75", "out27"], ["out76"]),
        (Layer45(), ["out76"], ["out77"]),
        (Layer46(), ["out77"], ["out78"]),
        (Layer47(), ["out75", "out78"], ["out79"]),
        (Layer48(), ["out79"], ["out81"]),
        (Layer49(), ["out81"], ["out82"]),
        (Layer50(), ["out82"], ["out83"]),
        (Layer51(), ["out83"], ["out84"]),
        (Layer52(), ["out81", "out84"], ["out85"]),
        (Layer53(), ["out85"], ["out87"]),
        (Layer54(), ["out27", "out87"], ["out88"]),
        (Layer55(), ["out88"], ["out89"]),
        (Layer56(), ["out89"], ["out90"]),
        (Layer57(), ["out87", "out90"], ["out91"]),
        (Layer58(), ["out91"], ["out93"]),
        (Layer59(), ["out93"], ["out94"]),
        (Layer60(), ["out94"], ["out95"]),
        (Layer61(), ["out95"], ["out96"]),
        (Layer62(), ["out93", "out96"], ["out97"]),
        (Layer63(), ["out97"], ["out99"]),
        (Layer64(), ["out27", "out99"], ["out100"]),
        (Layer65(), ["out100"], ["out101"]),
        (Layer66(), ["out101"], ["out102"]),
        (Layer67(), ["out99", "out102"], ["out103"]),
        (Layer68(), ["out103"], ["out105"]),
        (Layer69(), ["out105"], ["out106"]),
        (Layer70(), ["out106"], ["out107"]),
        (Layer71(), ["out107"], ["out108"]),
        (Layer72(), ["out105", "out108"], ["out109"]),
        (Layer73(), ["out109"], ["out111"]),
        (Layer74(), ["out27", "out111"], ["out112"]),
        (Layer75(), ["out112"], ["out113"]),
        (Layer76(), ["out113"], ["out114"]),
        (Layer77(), ["out111", "out114"], ["out115"]),
        (Layer78(), ["out115"], ["out117"]),
        (Layer79(), ["out117"], ["out118"]),
        (Layer80(), ["out118"], ["out119"]),
        (Layer81(), ["out119"], ["out120"]),
        (Layer82(), ["out117", "out120"], ["out121"]),
        (Layer83(), ["out121"], ["out123"]),
        (Layer84(), ["out27", "out123"], ["out124"]),
        (Layer85(), ["out124"], ["out125"]),
        (Layer86(), ["out125"], ["out126"]),
        (Layer87(), ["out123", "out126"], ["out127"]),
        (Layer88(), ["out127"], ["out129"]),
        (Layer89(), ["out129"], ["out130"]),
        (Layer90(), ["out130"], ["out131"]),
        (Layer91(), ["out131"], ["out132"]),
        (Layer92(), ["out129", "out132"], ["out133"]),
        (Layer93(), ["out133"], ["out135"]),
        (Layer94(), ["out27", "out135"], ["out136"]),
        (Layer95(), ["out136"], ["out137"]),
        (Layer96(), ["out137"], ["out138"]),
        (Layer97(), ["out135", "out138"], ["out139"]),
        (Layer98(), ["out139"], ["out141"]),
        (Layer99(), ["out141"], ["out142"]),
        (Layer100(), ["out142"], ["out143"]),
        (Layer101(), ["out143"], ["out144"]),
        (Layer102(), ["out141", "out144"], ["out145"]),
        (Layer103(), ["out145"], ["out147"]),
        (Layer104(), ["out147", "out27"], ["out148"]),
        (Layer105(), ["out148"], ["out149"]),
        (Layer106(), ["out149"], ["out150"]),
        (Layer107(), ["out147", "out150"], ["out151"]),
        (Layer108(), ["out151"], ["out153"]),
        (Layer109(), ["out153"], ["out154"]),
        (Layer110(), ["out154"], ["out155"]),
        (Layer111(), ["out155"], ["out156"]),
        (Layer112(), ["out153", "out156"], ["out157"]),
        (Layer113(), ["out157"], ["out159"]),
        (Layer114(), ["out159", "out27"], ["out160"]),
        (Layer115(), ["out160"], ["out161"]),
        (Layer116(), ["out161"], ["out162"]),
        (Layer117(), ["out159", "out162"], ["out163"]),
        (Layer118(), ["out163"], ["out165"]),
        (Layer119(), ["out165"], ["out166"]),
        (Layer120(), ["out166"], ["out167"]),
        (Layer121(), ["out167"], ["out168"]),
        (Layer122(), ["out165", "out168"], ["out169"]),
        (Layer123(), ["out169"], ["out171"]),
        (Layer124(), ["out171", "out27"], ["out172"]),
        (Layer125(), ["out172"], ["out173"]),
        (Layer126(), ["out173"], ["out174"]),
        (Layer127(), ["out171", "out174"], ["out175"]),
        (Layer128(), ["out175"], ["out177"]),
        (Layer129(), ["out177"], ["out178"]),
        (Layer130(), ["out178"], ["out179"]),
        (Layer131(), ["out179"], ["out180"]),
        (Layer132(), ["out177", "out180"], ["out181"]),
        (Layer133(), ["out181"], ["out183"]),
        (Layer134(), ["out183", "out27"], ["out184"]),
        (Layer135(), ["out184"], ["out185"]),
        (Layer136(), ["out185"], ["out186"]),
        (Layer137(), ["out183", "out186"], ["out187"]),
        (Layer138(), ["out187"], ["out189"]),
        (Layer139(), ["out189"], ["out190"]),
        (Layer140(), ["out190"], ["out191"]),
        (Layer141(), ["out191"], ["out192"]),
        (Layer142(), ["out189", "out192"], ["out193"]),
        (Layer143(), ["out193"], ["out195"]),
        (Layer144(), ["out195", "out27"], ["out196"]),
        (Layer145(), ["out196"], ["out197"]),
        (Layer146(), ["out197"], ["out198"]),
        (Layer147(), ["out195", "out198"], ["out199"]),
        (Layer148(), ["out199"], ["out201"]),
        (Layer149(), ["out201"], ["out202"]),
        (Layer150(), ["out202"], ["out203"]),
        (Layer151(), ["out203"], ["out204"]),
        (Layer152(), ["out201", "out204"], ["out205"]),
        (Layer153(), ["out205"], ["out207"]),
        (Layer154(), ["out207", "out27"], ["out208"]),
        (Layer155(), ["out208"], ["out209"]),
        (Layer156(), ["out209"], ["out210"]),
        (Layer157(), ["out207", "out210"], ["out211"]),
        (Layer158(), ["out211"], ["out213"]),
        (Layer159(), ["out213"], ["out214"]),
        (Layer160(), ["out214"], ["out215"]),
        (Layer161(), ["out215"], ["out216"]),
        (Layer162(), ["out213", "out216"], ["out217"]),
        (Layer163(), ["out217"], ["out219"]),
        (Layer164(), ["out219", "out27"], ["out220"]),
        (Layer165(), ["out220"], ["out221"]),
        (Layer166(), ["out221"], ["out222"]),
        (Layer167(), ["out219", "out222"], ["out223"]),
        (Layer168(), ["out223"], ["out225"]),
        (Layer169(), ["out225"], ["out226"]),
        (Layer170(), ["out226"], ["out227"]),
        (Layer171(), ["out227"], ["out228"]),
        (Layer172(), ["out225", "out228"], ["out229"]),
        (Layer173(), ["out229"], ["out231"]),
        (Layer174(), ["out231", "out27"], ["out232"]),
        (Layer175(), ["out232"], ["out233"]),
        (Layer176(), ["out233"], ["out234"]),
        (Layer177(), ["out231", "out234"], ["out235"]),
        (Layer178(), ["out235"], ["out237"]),
        (Layer179(), ["out237"], ["out238"]),
        (Layer180(), ["out238"], ["out239"]),
        (Layer181(), ["out239"], ["out240"]),
        (Layer182(), ["out237", "out240"], ["out241"]),
        (Layer183(), ["out241"], ["out243"]),
        (Layer184(), ["out243", "out27"], ["out244"]),
        (Layer185(), ["out244"], ["out245"]),
        (Layer186(), ["out245"], ["out246"]),
        (Layer187(), ["out243", "out246"], ["out247"]),
        (Layer188(), ["out247"], ["out249"]),
        (Layer189(), ["out249"], ["out250"]),
        (Layer190(), ["out250"], ["out251"]),
        (Layer191(), ["out251"], ["out252"]),
        (Layer192(), ["out249", "out252"], ["out253"]),
        (Layer193(), ["out253"], ["out255"]),
        (Layer194(), ["out255", "out27"], ["out256"]),
        (Layer195(), ["out256"], ["out257"]),
        (Layer196(), ["out257"], ["out258"]),
        (Layer197(), ["out255", "out258"], ["out259"]),
        (Layer198(), ["out259"], ["out261"]),
        (Layer199(), ["out261"], ["out262"]),
        (Layer200(), ["out262"], ["out263"]),
        (Layer201(), ["out263"], ["out264"]),
        (Layer202(), ["out261", "out264"], ["out265"]),
        (Layer203(), ["out265"], ["out267"]),
        (Layer204(), ["out267", "out27"], ["out268"]),
        (Layer205(), ["out268"], ["out269"]),
        (Layer206(), ["out269"], ["out270"]),
        (Layer207(), ["out267", "out270"], ["out271"]),
        (Layer208(), ["out271"], ["out273"]),
        (Layer209(), ["out273"], ["out274"]),
        (Layer210(), ["out274"], ["out275"]),
        (Layer211(), ["out275"], ["out276"]),
        (Layer212(), ["out273", "out276"], ["out277"]),
        (Layer213(), ["out277"], ["out279"]),
        (Layer214(), ["out279", "out27"], ["out280"]),
        (Layer215(), ["out280"], ["out281"]),
        (Layer216(), ["out281"], ["out282"]),
        (Layer217(), ["out279", "out282"], ["out283"]),
        (Layer218(), ["out283"], ["out285"]),
        (Layer219(), ["out285"], ["out286"]),
        (Layer220(), ["out286"], ["out287"]),
        (Layer221(), ["out287"], ["out288"]),
        (Layer222(), ["out285", "out288"], ["out289"]),
        (Layer223(), ["out289"], ["out291"]),
        (Layer224(), ["out291", "out27"], ["out292"]),
        (Layer225(), ["out292"], ["out293"]),
        (Layer226(), ["out293"], ["out294"]),
        (Layer227(), ["out291", "out294"], ["out295"]),
        (Layer228(), ["out295"], ["out297"]),
        (Layer229(), ["out297"], ["out298"]),
        (Layer230(), ["out298"], ["out299"]),
        (Layer231(), ["out299"], ["out300"]),
        (Layer232(), ["out297", "out300"], ["out301"]),
        (Layer233(), ["out301"], ["out303"]),
        (Layer234(), ["out303", "out27"], ["out304"]),
        (Layer235(), ["out304"], ["out305"]),
        (Layer236(), ["out305"], ["out306"]),
        (Layer237(), ["out303", "out306"], ["out307"]),
        (Layer238(), ["out307"], ["out309"]),
        (Layer239(), ["out309"], ["out310"]),
        (Layer240(), ["out310"], ["out311"]),
        (Layer241(), ["out311"], ["out312"]),
        (Layer242(), ["out309", "out312"], ["out313"]),
        (Layer243(), ["out313"], ["out315"]),
        (Layer244(), ["out315"], ["out316"]),
        (Layer245(), ["out315"], ["out317"]),
        (Layer246(), ["out317"], ["out318"]),
        (Layer247(), ["out318"], ["out319"]),
        (Layer248(), ["out319"], ["out320"]),
        (Layer249(), ["out316"], ["out321"]),
        (criterion, ["out321"], ["loss"])
    ]

class Layer2(torch.nn.Module):
    def __init__(self):
        super(Layer2, self).__init__()
        self.layer0 = BertEmbeddings(30528, 1024, 512, 2, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer4(torch.nn.Module):
    def __init__(self):
        super(Layer4, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer5(torch.nn.Module):
    def __init__(self):
        super(Layer5, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer6(torch.nn.Module):
    def __init__(self):
        super(Layer6, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer7(torch.nn.Module):
    def __init__(self):
        super(Layer7, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer8(torch.nn.Module):
    def __init__(self):
        super(Layer8, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer9(torch.nn.Module):
    def __init__(self):
        super(Layer9, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer10(torch.nn.Module):
    def __init__(self):
        super(Layer10, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer11(torch.nn.Module):
    def __init__(self):
        super(Layer11, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer12(torch.nn.Module):
    def __init__(self):
        super(Layer12, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer13(torch.nn.Module):
    def __init__(self):
        super(Layer13, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer14(torch.nn.Module):
    def __init__(self):
        super(Layer14, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer15(torch.nn.Module):
    def __init__(self):
        super(Layer15, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer16(torch.nn.Module):
    def __init__(self):
        super(Layer16, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer17(torch.nn.Module):
    def __init__(self):
        super(Layer17, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer18(torch.nn.Module):
    def __init__(self):
        super(Layer18, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer19(torch.nn.Module):
    def __init__(self):
        super(Layer19, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer20(torch.nn.Module):
    def __init__(self):
        super(Layer20, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer21(torch.nn.Module):
    def __init__(self):
        super(Layer21, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer22(torch.nn.Module):
    def __init__(self):
        super(Layer22, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer23(torch.nn.Module):
    def __init__(self):
        super(Layer23, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer24(torch.nn.Module):
    def __init__(self):
        super(Layer24, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer25(torch.nn.Module):
    def __init__(self):
        super(Layer25, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer26(torch.nn.Module):
    def __init__(self):
        super(Layer26, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer27(torch.nn.Module):
    def __init__(self):
        super(Layer27, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer28(torch.nn.Module):
    def __init__(self):
        super(Layer28, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer29(torch.nn.Module):
    def __init__(self):
        super(Layer29, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer30(torch.nn.Module):
    def __init__(self):
        super(Layer30, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer31(torch.nn.Module):
    def __init__(self):
        super(Layer31, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer32(torch.nn.Module):
    def __init__(self):
        super(Layer32, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer33(torch.nn.Module):
    def __init__(self):
        super(Layer33, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer34(torch.nn.Module):
    def __init__(self):
        super(Layer34, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer35(torch.nn.Module):
    def __init__(self):
        super(Layer35, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer36(torch.nn.Module):
    def __init__(self):
        super(Layer36, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer37(torch.nn.Module):
    def __init__(self):
        super(Layer37, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer38(torch.nn.Module):
    def __init__(self):
        super(Layer38, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer39(torch.nn.Module):
    def __init__(self):
        super(Layer39, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer40(torch.nn.Module):
    def __init__(self):
        super(Layer40, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer41(torch.nn.Module):
    def __init__(self):
        super(Layer41, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer42(torch.nn.Module):
    def __init__(self):
        super(Layer42, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer43(torch.nn.Module):
    def __init__(self):
        super(Layer43, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer44(torch.nn.Module):
    def __init__(self):
        super(Layer44, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer45(torch.nn.Module):
    def __init__(self):
        super(Layer45, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer46(torch.nn.Module):
    def __init__(self):
        super(Layer46, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer47(torch.nn.Module):
    def __init__(self):
        super(Layer47, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer48(torch.nn.Module):
    def __init__(self):
        super(Layer48, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer49(torch.nn.Module):
    def __init__(self):
        super(Layer49, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer50(torch.nn.Module):
    def __init__(self):
        super(Layer50, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer51(torch.nn.Module):
    def __init__(self):
        super(Layer51, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer52(torch.nn.Module):
    def __init__(self):
        super(Layer52, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer53(torch.nn.Module):
    def __init__(self):
        super(Layer53, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer54(torch.nn.Module):
    def __init__(self):
        super(Layer54, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input1, input0):
        out0 = self.layer0(input0, input1)
        return out0

class Layer55(torch.nn.Module):
    def __init__(self):
        super(Layer55, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer56(torch.nn.Module):
    def __init__(self):
        super(Layer56, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer57(torch.nn.Module):
    def __init__(self):
        super(Layer57, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer58(torch.nn.Module):
    def __init__(self):
        super(Layer58, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer59(torch.nn.Module):
    def __init__(self):
        super(Layer59, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer60(torch.nn.Module):
    def __init__(self):
        super(Layer60, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer61(torch.nn.Module):
    def __init__(self):
        super(Layer61, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer62(torch.nn.Module):
    def __init__(self):
        super(Layer62, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer63(torch.nn.Module):
    def __init__(self):
        super(Layer63, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer64(torch.nn.Module):
    def __init__(self):
        super(Layer64, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input1, input0):
        out0 = self.layer0(input0, input1)
        return out0

class Layer65(torch.nn.Module):
    def __init__(self):
        super(Layer65, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer66(torch.nn.Module):
    def __init__(self):
        super(Layer66, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer67(torch.nn.Module):
    def __init__(self):
        super(Layer67, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer68(torch.nn.Module):
    def __init__(self):
        super(Layer68, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer69(torch.nn.Module):
    def __init__(self):
        super(Layer69, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer70(torch.nn.Module):
    def __init__(self):
        super(Layer70, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer71(torch.nn.Module):
    def __init__(self):
        super(Layer71, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer72(torch.nn.Module):
    def __init__(self):
        super(Layer72, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer73(torch.nn.Module):
    def __init__(self):
        super(Layer73, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer74(torch.nn.Module):
    def __init__(self):
        super(Layer74, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input1, input0):
        out0 = self.layer0(input0, input1)
        return out0

class Layer75(torch.nn.Module):
    def __init__(self):
        super(Layer75, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer76(torch.nn.Module):
    def __init__(self):
        super(Layer76, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer77(torch.nn.Module):
    def __init__(self):
        super(Layer77, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer78(torch.nn.Module):
    def __init__(self):
        super(Layer78, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer79(torch.nn.Module):
    def __init__(self):
        super(Layer79, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer80(torch.nn.Module):
    def __init__(self):
        super(Layer80, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer81(torch.nn.Module):
    def __init__(self):
        super(Layer81, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer82(torch.nn.Module):
    def __init__(self):
        super(Layer82, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer83(torch.nn.Module):
    def __init__(self):
        super(Layer83, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer84(torch.nn.Module):
    def __init__(self):
        super(Layer84, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input1, input0):
        out0 = self.layer0(input0, input1)
        return out0

class Layer85(torch.nn.Module):
    def __init__(self):
        super(Layer85, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer86(torch.nn.Module):
    def __init__(self):
        super(Layer86, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer87(torch.nn.Module):
    def __init__(self):
        super(Layer87, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer88(torch.nn.Module):
    def __init__(self):
        super(Layer88, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer89(torch.nn.Module):
    def __init__(self):
        super(Layer89, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer90(torch.nn.Module):
    def __init__(self):
        super(Layer90, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer91(torch.nn.Module):
    def __init__(self):
        super(Layer91, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer92(torch.nn.Module):
    def __init__(self):
        super(Layer92, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer93(torch.nn.Module):
    def __init__(self):
        super(Layer93, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer94(torch.nn.Module):
    def __init__(self):
        super(Layer94, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input1, input0):
        out0 = self.layer0(input0, input1)
        return out0

class Layer95(torch.nn.Module):
    def __init__(self):
        super(Layer95, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer96(torch.nn.Module):
    def __init__(self):
        super(Layer96, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer97(torch.nn.Module):
    def __init__(self):
        super(Layer97, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer98(torch.nn.Module):
    def __init__(self):
        super(Layer98, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer99(torch.nn.Module):
    def __init__(self):
        super(Layer99, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer100(torch.nn.Module):
    def __init__(self):
        super(Layer100, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer101(torch.nn.Module):
    def __init__(self):
        super(Layer101, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer102(torch.nn.Module):
    def __init__(self):
        super(Layer102, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer103(torch.nn.Module):
    def __init__(self):
        super(Layer103, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer104(torch.nn.Module):
    def __init__(self):
        super(Layer104, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer105(torch.nn.Module):
    def __init__(self):
        super(Layer105, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer106(torch.nn.Module):
    def __init__(self):
        super(Layer106, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer107(torch.nn.Module):
    def __init__(self):
        super(Layer107, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer108(torch.nn.Module):
    def __init__(self):
        super(Layer108, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer109(torch.nn.Module):
    def __init__(self):
        super(Layer109, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer110(torch.nn.Module):
    def __init__(self):
        super(Layer110, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer111(torch.nn.Module):
    def __init__(self):
        super(Layer111, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer112(torch.nn.Module):
    def __init__(self):
        super(Layer112, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer113(torch.nn.Module):
    def __init__(self):
        super(Layer113, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer114(torch.nn.Module):
    def __init__(self):
        super(Layer114, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer115(torch.nn.Module):
    def __init__(self):
        super(Layer115, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer116(torch.nn.Module):
    def __init__(self):
        super(Layer116, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer117(torch.nn.Module):
    def __init__(self):
        super(Layer117, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer118(torch.nn.Module):
    def __init__(self):
        super(Layer118, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer119(torch.nn.Module):
    def __init__(self):
        super(Layer119, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer120(torch.nn.Module):
    def __init__(self):
        super(Layer120, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer121(torch.nn.Module):
    def __init__(self):
        super(Layer121, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer122(torch.nn.Module):
    def __init__(self):
        super(Layer122, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer123(torch.nn.Module):
    def __init__(self):
        super(Layer123, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer124(torch.nn.Module):
    def __init__(self):
        super(Layer124, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer125(torch.nn.Module):
    def __init__(self):
        super(Layer125, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer126(torch.nn.Module):
    def __init__(self):
        super(Layer126, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer127(torch.nn.Module):
    def __init__(self):
        super(Layer127, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer128(torch.nn.Module):
    def __init__(self):
        super(Layer128, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer129(torch.nn.Module):
    def __init__(self):
        super(Layer129, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer130(torch.nn.Module):
    def __init__(self):
        super(Layer130, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer131(torch.nn.Module):
    def __init__(self):
        super(Layer131, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer132(torch.nn.Module):
    def __init__(self):
        super(Layer132, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer133(torch.nn.Module):
    def __init__(self):
        super(Layer133, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer134(torch.nn.Module):
    def __init__(self):
        super(Layer134, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer135(torch.nn.Module):
    def __init__(self):
        super(Layer135, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer136(torch.nn.Module):
    def __init__(self):
        super(Layer136, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer137(torch.nn.Module):
    def __init__(self):
        super(Layer137, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer138(torch.nn.Module):
    def __init__(self):
        super(Layer138, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer139(torch.nn.Module):
    def __init__(self):
        super(Layer139, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer140(torch.nn.Module):
    def __init__(self):
        super(Layer140, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer141(torch.nn.Module):
    def __init__(self):
        super(Layer141, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer142(torch.nn.Module):
    def __init__(self):
        super(Layer142, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer143(torch.nn.Module):
    def __init__(self):
        super(Layer143, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer144(torch.nn.Module):
    def __init__(self):
        super(Layer144, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer145(torch.nn.Module):
    def __init__(self):
        super(Layer145, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer146(torch.nn.Module):
    def __init__(self):
        super(Layer146, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer147(torch.nn.Module):
    def __init__(self):
        super(Layer147, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer148(torch.nn.Module):
    def __init__(self):
        super(Layer148, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer149(torch.nn.Module):
    def __init__(self):
        super(Layer149, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer150(torch.nn.Module):
    def __init__(self):
        super(Layer150, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer151(torch.nn.Module):
    def __init__(self):
        super(Layer151, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer152(torch.nn.Module):
    def __init__(self):
        super(Layer152, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer153(torch.nn.Module):
    def __init__(self):
        super(Layer153, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer154(torch.nn.Module):
    def __init__(self):
        super(Layer154, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer155(torch.nn.Module):
    def __init__(self):
        super(Layer155, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer156(torch.nn.Module):
    def __init__(self):
        super(Layer156, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer157(torch.nn.Module):
    def __init__(self):
        super(Layer157, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer158(torch.nn.Module):
    def __init__(self):
        super(Layer158, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer159(torch.nn.Module):
    def __init__(self):
        super(Layer159, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer160(torch.nn.Module):
    def __init__(self):
        super(Layer160, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer161(torch.nn.Module):
    def __init__(self):
        super(Layer161, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer162(torch.nn.Module):
    def __init__(self):
        super(Layer162, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer163(torch.nn.Module):
    def __init__(self):
        super(Layer163, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer164(torch.nn.Module):
    def __init__(self):
        super(Layer164, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer165(torch.nn.Module):
    def __init__(self):
        super(Layer165, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer166(torch.nn.Module):
    def __init__(self):
        super(Layer166, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer167(torch.nn.Module):
    def __init__(self):
        super(Layer167, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer168(torch.nn.Module):
    def __init__(self):
        super(Layer168, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer169(torch.nn.Module):
    def __init__(self):
        super(Layer169, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer170(torch.nn.Module):
    def __init__(self):
        super(Layer170, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer171(torch.nn.Module):
    def __init__(self):
        super(Layer171, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer172(torch.nn.Module):
    def __init__(self):
        super(Layer172, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer173(torch.nn.Module):
    def __init__(self):
        super(Layer173, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer174(torch.nn.Module):
    def __init__(self):
        super(Layer174, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer175(torch.nn.Module):
    def __init__(self):
        super(Layer175, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer176(torch.nn.Module):
    def __init__(self):
        super(Layer176, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer177(torch.nn.Module):
    def __init__(self):
        super(Layer177, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer178(torch.nn.Module):
    def __init__(self):
        super(Layer178, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer179(torch.nn.Module):
    def __init__(self):
        super(Layer179, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer180(torch.nn.Module):
    def __init__(self):
        super(Layer180, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer181(torch.nn.Module):
    def __init__(self):
        super(Layer181, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer182(torch.nn.Module):
    def __init__(self):
        super(Layer182, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer183(torch.nn.Module):
    def __init__(self):
        super(Layer183, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer184(torch.nn.Module):
    def __init__(self):
        super(Layer184, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer185(torch.nn.Module):
    def __init__(self):
        super(Layer185, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer186(torch.nn.Module):
    def __init__(self):
        super(Layer186, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer187(torch.nn.Module):
    def __init__(self):
        super(Layer187, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer188(torch.nn.Module):
    def __init__(self):
        super(Layer188, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer189(torch.nn.Module):
    def __init__(self):
        super(Layer189, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer190(torch.nn.Module):
    def __init__(self):
        super(Layer190, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer191(torch.nn.Module):
    def __init__(self):
        super(Layer191, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer192(torch.nn.Module):
    def __init__(self):
        super(Layer192, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer193(torch.nn.Module):
    def __init__(self):
        super(Layer193, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer194(torch.nn.Module):
    def __init__(self):
        super(Layer194, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer195(torch.nn.Module):
    def __init__(self):
        super(Layer195, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer196(torch.nn.Module):
    def __init__(self):
        super(Layer196, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer197(torch.nn.Module):
    def __init__(self):
        super(Layer197, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer198(torch.nn.Module):
    def __init__(self):
        super(Layer198, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer199(torch.nn.Module):
    def __init__(self):
        super(Layer199, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer200(torch.nn.Module):
    def __init__(self):
        super(Layer200, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer201(torch.nn.Module):
    def __init__(self):
        super(Layer201, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer202(torch.nn.Module):
    def __init__(self):
        super(Layer202, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer203(torch.nn.Module):
    def __init__(self):
        super(Layer203, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer204(torch.nn.Module):
    def __init__(self):
        super(Layer204, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer205(torch.nn.Module):
    def __init__(self):
        super(Layer205, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer206(torch.nn.Module):
    def __init__(self):
        super(Layer206, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer207(torch.nn.Module):
    def __init__(self):
        super(Layer207, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer208(torch.nn.Module):
    def __init__(self):
        super(Layer208, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer209(torch.nn.Module):
    def __init__(self):
        super(Layer209, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer210(torch.nn.Module):
    def __init__(self):
        super(Layer210, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer211(torch.nn.Module):
    def __init__(self):
        super(Layer211, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer212(torch.nn.Module):
    def __init__(self):
        super(Layer212, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer213(torch.nn.Module):
    def __init__(self):
        super(Layer213, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer214(torch.nn.Module):
    def __init__(self):
        super(Layer214, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer215(torch.nn.Module):
    def __init__(self):
        super(Layer215, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer216(torch.nn.Module):
    def __init__(self):
        super(Layer216, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer217(torch.nn.Module):
    def __init__(self):
        super(Layer217, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer218(torch.nn.Module):
    def __init__(self):
        super(Layer218, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer219(torch.nn.Module):
    def __init__(self):
        super(Layer219, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer220(torch.nn.Module):
    def __init__(self):
        super(Layer220, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer221(torch.nn.Module):
    def __init__(self):
        super(Layer221, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer222(torch.nn.Module):
    def __init__(self):
        super(Layer222, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer223(torch.nn.Module):
    def __init__(self):
        super(Layer223, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer224(torch.nn.Module):
    def __init__(self):
        super(Layer224, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer225(torch.nn.Module):
    def __init__(self):
        super(Layer225, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer226(torch.nn.Module):
    def __init__(self):
        super(Layer226, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer227(torch.nn.Module):
    def __init__(self):
        super(Layer227, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer228(torch.nn.Module):
    def __init__(self):
        super(Layer228, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer229(torch.nn.Module):
    def __init__(self):
        super(Layer229, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer230(torch.nn.Module):
    def __init__(self):
        super(Layer230, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer231(torch.nn.Module):
    def __init__(self):
        super(Layer231, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer232(torch.nn.Module):
    def __init__(self):
        super(Layer232, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer233(torch.nn.Module):
    def __init__(self):
        super(Layer233, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer234(torch.nn.Module):
    def __init__(self):
        super(Layer234, self).__init__()
        self.layer0 = BertSelfAttention(1024, 16, 0.1)

    def forward(self, input0, input1):
        out0 = self.layer0(input0, input1)
        return out0

class Layer235(torch.nn.Module):
    def __init__(self):
        super(Layer235, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer236(torch.nn.Module):
    def __init__(self):
        super(Layer236, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer237(torch.nn.Module):
    def __init__(self):
        super(Layer237, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer238(torch.nn.Module):
    def __init__(self):
        super(Layer238, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer239(torch.nn.Module):
    def __init__(self):
        super(Layer239, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=4096, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer240(torch.nn.Module):
    def __init__(self):
        super(Layer240, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer241(torch.nn.Module):
    def __init__(self):
        super(Layer241, self).__init__()
        self.layer0 = torch.nn.Dropout(p=0.1)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer242(torch.nn.Module):
    def __init__(self):
        super(Layer242, self).__init__()
        

    def forward(self, input1, input0):
        input0 = input0 + input1
        return input0

class Layer243(torch.nn.Module):
    def __init__(self):
        super(Layer243, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer244(torch.nn.Module):
    def __init__(self):
        super(Layer244, self).__init__()
        self.layer0 = BertPooler(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer245(torch.nn.Module):
    def __init__(self):
        super(Layer245, self).__init__()
        self.layer0 = LinearActivation(in_features=1024, out_features=1024, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer246(torch.nn.Module):
    def __init__(self):
        super(Layer246, self).__init__()
        self.layer0 = BertLayerNorm(1024)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer247(torch.nn.Module):
    def __init__(self):
        super(Layer247, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=30528, bias=False)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer248(torch.nn.Module):
    def __init__(self):
        super(Layer248, self).__init__()
        self.layer0 = BertAdd(30528)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0

class Layer249(torch.nn.Module):
    def __init__(self):
        super(Layer249, self).__init__()
        self.layer0 = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    def forward(self, input0):
        out0 = self.layer0(input0)
        return out0