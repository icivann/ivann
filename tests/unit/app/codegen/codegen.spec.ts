import generateCode from '@/app/codegen/codeGenerator';
import istateToGraph from '@/app/ir/istateToGraph';

function removeBlankLines(x: string): string {
  return x.trim();
}
const conv2dDef = '{"nodes":[{"type":"I/O","id":"node_16039892032880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039892032881","value":null}]],"position":{"x":21,"y":234},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16039892157612","name":"Conv2d","options":[["In channels", 16],["Out channels", 32],["Kernel size", [2,2]],["Stride", [2,2]],["Padding", [2,2]],["Dilation", [2,2]],["Groups", 1],["Bias", 1],["Padding mode", "reflect"]],"state":{},"interfaces":[["Input",{"id":"ni_16039892157623","value":null}],["Output",{"id":"ni_16039892157624","value":null}]],"position":{"x":312,"y":230},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_16039892316758","name":"MaxPool2d","options":['
  + '["Kernel size", [0,0]],["Stride", [1,1]],["Padding", [1,1]],["Dilation",[1,1]],["Return indices", null],["Ceil mode", null]],'
  + '"state":{},"interfaces":[["Input",{"id":"ni_16039892316759","value":null}],["Output",{"id":"ni_160398923167510","value":null}]],"position":{"x":331,"y":479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160398924063014","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160398924063015","value":null}]],"position":{"x":608,"y":437},"width":200,"twoColumn":false}],"connections":[{"id":"16039892205757","from":"ni_16039892032881","to":"ni_16039892157623"},{"id":"160398923656513","from":"ni_16039892157624","to":"ni_16039892316759"},{"id":"160398961820424","from":"ni_160398923167510","to":"ni_160398924063015"}],"panning":{"x":0,"y":0},"scaling":1}';

const conv2dDefGenerated = 'nn.Conv2d(in_channels=0, out_channels=0, kernel_size=(0,0), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=1, padding_mode=\'zeros\')';
const maxpool2dDefGenerated = 'nn.MaxPool2d(kernel_size=(0,0), stride=(0,0), padding=(0,0), dilation=(1,1), return_indices=null, ceil_mode=null)';

describe('codegen', () => {
  it('renders props.msg when passed', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":31,"y":162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":245,"y":150},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513148090922","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513148090923","value":null}],["Output",{"id":"ni_160513148090924","value":null}]],"position":{"x":454,"y":163},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513148489227","from":"ni_16051304343279","to":"ni_160513148090923"},{"id":"160513148671330","from":"ni_160513148090924","to":"ni_160513044748320"}],"panning":{"x":0,"y":0},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, inmodel_1):
    x = inmodel_1
    x = self.conv2d_1(x)
    x = self.maxpool2d_1(x)
    return x`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for concat node', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":31,"y":162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":245,"y":150},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513148090922","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513148090923","value":null}],["Output",{"id":"ni_160513148090924","value":null}]],"position":{"x":251,"y":582},"width":200,"twoColumn":false},{"type":"Concat","id":"node_160513194994734","name":"Concat","options":[],"state":{},"interfaces":[["Input 1",{"id":"ni_160513194994835","value":null}],["Input 2",{"id":"ni_160513194994836","value":null}],["Output",{"id":"ni_160513194994837","value":null}]],"position":{"x":493,"y":274},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513194393533","from":"ni_16051304291466","to":"ni_160513148090923"},{"id":"160513195310840","from":"ni_16051304343279","to":"ni_160513194994835"},{"id":"160513195526843","from":"ni_160513148090924","to":"ni_160513194994836"},{"id":"160513195733046","from":"ni_160513194994837","to":"ni_160513044748320"}],"panning":{"x":14,"y":-145},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, inmodel_1):
    x = inmodel_1
    x_1 = self.conv2d_1(x)
    x_2 = self.maxpool2d_1(x)
    x_3 = torch.cat(x_1, x_2)
    return x_3`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for nodes with two outputs', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":-67,"y":163},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":161,"y":147},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513234026953","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513234026954","value":null}],["Output",{"id":"ni_160513234026955","value":null}]],"position":{"x":394,"y":120},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513234536963","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513234536964","value":null}],["Output",{"id":"ni_160513234536965","value":null}]],"position":{"x":414,"y":314},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513235923569","name":"output2","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513235923570","value":null}]],"position":{"x":734,"y":329},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513234209059","from":"ni_16051304343279","to":"ni_160513234026954"},{"id":"160513234367862","from":"ni_160513234026955","to":"ni_160513044748320"},{"id":"160513234901968","from":"ni_16051304343279","to":"ni_160513234536964"},{"id":"160513236247273","from":"ni_160513234536965","to":"ni_160513235923570"}],"panning":{"x":86,"y":-41},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = ${conv2dDefGenerated}
    self.conv2d_2 = ${conv2dDefGenerated}
    self.conv2d_3 = ${conv2dDefGenerated}

  def forward(self, inmodel_1):
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = self.conv2d_2(x)
    x_2 = self.conv2d_3(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for nested branching', () => {
    const iState = JSON.parse(
      '{"nodes":[{"type":"I/O","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":-101.39562776264694,"y":133.61712921026987},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16039996941842","name":"Conv2d","options":[["In_Channels",16],["Out_Channels",32],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":178.07554955335334,"y":141.4143731668673},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400104763047","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400104763048","value":null}],["Output",{"id":"ni_160400104763149","value":null}]],"position":{"x":503.7178227039244,"y":59.36506071686672},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400110066953","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400110066954","value":null}],["Output",{"id":"ni_160400110066955","value":null}]],"position":{"x":508.5378485872821,"y":288.9421610499153},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400112448859","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400112448860","value":null}]],"position":{"x":911.0666462044529,"y":29.87844102632479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400113171465","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400113171466","value":null}]],"position":{"x":939.5004691338713,"y":306.3183861734489},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160400907253476","name":"MaxPool2d","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_160400907253477","value":null}],["Output",{"id":"ni_160400907253578","value":null}]],"position":{"x":467.03535191811665,"y":543.8685871785077},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400908547183","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400908547284","value":null}],["Output",{"id":"ni_160400908547285","value":null}]],"position":{"x":861.0397286512999,"y":536.5040193891028},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400909098789","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400909098890","value":null}],["Output",{"id":"ni_160400909098891","value":null}]],"position":{"x":849.9928769671922,"y":751.9176272292082},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400911038095","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400911038196","value":null}]],"position":{"x":1232.4724805786134,"y":514.4538965152768},"width":200,"twoColumn":false},{"type":"I/O","id":"node_1604009132060100","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1604009132060101","value":null}]],"position":{"x":1228.1276071453376,"y":707.8007642960396},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160400105094252","from":"ni_16039996941854","to":"ni_160400104763048"},{"id":"160400110318058","from":"ni_16039996941854","to":"ni_160400110066954"},{"id":"160400112847964","from":"ni_160400104763149","to":"ni_160400112448860"},{"id":"160400641665775","from":"ni_160400110066955","to":"ni_160400113171466"},{"id":"160400908003482","from":"ni_16039996941854","to":"ni_160400907253477"},{"id":"160400908905288","from":"ni_160400907253578","to":"ni_160400908547284"},{"id":"160400909363194","from":"ni_160400907253578","to":"ni_160400909098890"},{"id":"160400911382599","from":"ni_160400908547285","to":"ni_160400911038196"},{"id":"1604009155725107","from":"ni_160400909098891","to":"ni_1604009132060101"}],"panning":{"x":372.9582530167131,"y":321.9342393902725},"scaling":0.46031260305141036}',
    );
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = ${conv2dDefGenerated}
    self.conv2d_2 = ${conv2dDefGenerated}
    self.conv2d_3 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}
    self.conv2d_4 = ${conv2dDefGenerated}
    self.conv2d_5 = ${conv2dDefGenerated}

  def forward(self, inmodel_1):
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = self.conv2d_2(x)
    x_2 = self.conv2d_3(x)
    x_3 = self.maxpool2d_1(x)
    x_4 = self.conv2d_4(x_3)
    x_5 = self.conv2d_5(x_3)
    return x_1, x_2, x_4, x_5`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('branching input', () => {
    const iState = JSON.parse(
      '{"nodes":[{"type":"I/O","id":"node_16040096081900","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040096081911","value":null}]],"position":{"x":40,"y":182},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040096333102","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16040096333103","value":null}],["Output",{"id":"ni_16040096333104","value":null}]],"position":{"x":359,"y":152},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160400966555215","name":"MaxPool2d","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_160400966555216","value":null}],["Output",{"id":"ni_160400966555217","value":null}]],"position":{"x":375,"y":335},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400969358118","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400969358219","value":null}]],"position":{"x":891,"y":158},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400969919323","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400969919324","value":null}]],"position":{"x":868,"y":318},"width":200,"twoColumn":false}],"connections":[{"id":"16040096366997","from":"ni_16040096081911","to":"ni_16040096333103"},{"id":"160400969689622","from":"ni_16040096333104","to":"ni_160400969358219"},{"id":"160400970457030","from":"ni_16040096081911","to":"ni_160400966555216"},{"id":"160400971942533","from":"ni_160400966555217","to":"ni_160400969919324"}],"panning":{"x":0,"y":0},"scaling":1}',
    );
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = nn.Conv2d(16, 32, 2,2)
    self.maxpool2d_1 = nn.MaxPool2d((28,28))

  def forward(self, inmodel_1):
    x = inmodel_1
    x_1 = self.conv2d_1(x)
    x_2 = self.maxpool2d_1(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for custom nodes with a two parameters', () => {
    const iState = JSON.parse(
      '{"nodes":[{"type":"I/O","id":"node_16040522983880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040522983881","value":null}]],"position":{"x":61,"y":135},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040523365809","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160405233658010","value":null}],["Output",{"id":"ni_160405233658011","value":null}]],"position":{"x":315,"y":114},"width":200,"twoColumn":false},{"type":"Custom","id":"node_160405236950918","name":"Custom","options":[["Inline Code",{"text":"def customFunc(arg1):\\n  pass","hasError":false}]],"state":{},"interfaces":[["Output",{"id":"ni_160405236950919","value":null}],["arg1",{"id":"ni_160405238588426","value":null}]],"position":{"x":614,"y":116},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160405239680931","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160405239680932","value":null}]],"position":{"x":928,"y":183},"width":200,"twoColumn":false}],"connections":[{"id":"160405236380317","from":"ni_16040522983881","to":"ni_160405233658010"},{"id":"160405239983735","from":"ni_160405233658011","to":"ni_160405238588426"},{"id":"160405240249838","from":"ni_160405236950919","to":"ni_160405239680932"}],"panning":{"x":0,"y":0},"scaling":1}',
    );
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

def customFunc(arg1):
  pass

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = nn.Conv2d(16, 32, 2,2)

  def forward(self, inmodel_1):
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = customFunc(x)
    return x_1`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for custom nodes with a two parameters', () => {
    const iState = JSON.parse(conv2dDef);
    // '{"nodes":[{"type":"I/O","id":"node_16040522983880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040522983881","value":null}]],"position":{"x":61,"y":135},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040523365809","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160405233658010","value":null}],["Output",{"id":"ni_160405233658011","value":null}]],"position":{"x":315,"y":114},"width":200,"twoColumn":false},{"type":"Custom","id":"node_160405236950918","name":"Custom","options":[["Inline Code",{"text":"def customFunc(arg1):\\n  pass","hasError":false}]],"state":{},"interfaces":[["Output",{"id":"ni_160405236950919","value":null}],["arg1",{"id":"ni_160405238588426","value":null}]],"position":{"x":614,"y":116},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160405239680931","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160405239680932","value":null}]],"position":{"x":928,"y":183},"width":200,"twoColumn":false}],"connections":[{"id":"160405236380317","from":"ni_16040522983881","to":"ni_160405233658010"},{"id":"160405239983735","from":"ni_160405233658011","to":"ni_160405238588426"},{"id":"160405240249838","from":"ni_160405236950919","to":"ni_160405239680932"}],"panning":{"x":0,"y":0},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

def customFunc(arg1):
  pass

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = nn.Conv2d(16, 32, 2,2)

  def forward(self, inmodel_1):
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = customFunc(x)
    return x_1`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for custom nodes with a two parameters', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16040603729620","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040603729621","value":null}]],"position":{"x":35,"y":131},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040603828092","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16040603828093","value":null}],["Output",{"id":"ni_16040603828094","value":null}]],"position":{"x":328.8565635287275,"y":9.337310507178465},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040603862095","name":"Conv2d","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16040603862096","value":null}],["Output",{"id":"ni_16040603862097","value":null}]],"position":{"x":327.2333782717853,"y":212.2354676249726},"width":200,"twoColumn":false},{"type":"Custom","id":"node_16040603916568","name":"Custom","options":[["Inline Code",{"text":"def customFunc(arg1, arg2):\\n  pass","hasError":false}]],"state":{},"interfaces":[["Output",{"id":"ni_16040603916579","value":null}],["arg1",{"id":"ni_160406040197114","value":null}],["arg2",{"id":"ni_160406040197215","value":null}]],"position":{"x":642.1313181186007,"y":82.38064706958437},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160406040635116","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160406040635117","value":null}]],"position":{"x":1018.7102977292274,"y":105.10524066677729},"width":200,"twoColumn":false}],"connections":[{"id":"160406040940520","from":"ni_16040603729621","to":"ni_16040603828093"},{"id":"160406041213623","from":"ni_16040603729621","to":"ni_16040603862096"},{"id":"160406041415926","from":"ni_16040603828094","to":"ni_160406040197114"},{"id":"160406041624529","from":"ni_16040603862097","to":"ni_160406040197215"},{"id":"160406042274832","from":"ni_16040603916579","to":"ni_160406040635117"}],"panning":{"x":255.49012897051978,"y":102.66247222184393},"scaling":0.6160726237026897}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateCode(graph);
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F

def customFunc(arg1, arg2):
  pass

class Model(nn.Module):

  def __init__(self):
    self.conv2d_1 = nn.Conv2d(16, 32, 2,2)
    self.conv2d_2 = nn.Conv2d(16, 32, 2,2)

  def forward(self, inmodel_1):
    x = inmodel_1
    x_1 = self.conv2d_1(x)
    x_2 = self.conv2d_2(x)
    x_3 = customFunc(x_1, x_2)
    return x_3`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });
});
