import generateCode, { generateModelCode } from '@/app/codegen/codeGenerator';
import istateToGraph from '@/app/ir/istateToGraph';
import conv2d from '@/app/ir/model/conv2d';

function removeBlankLines(x: string): string {
  return x.trim();
}
const conv2dDef = '{"nodes":[{"type":"I/O","id":"node_16039892032880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039892032881","value":null}]],"position":{"x":21,"y":234},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16039892157612","name":"Conv2d","options":[["In channels", 16],["Out channels", 32],["Kernel size", [2,2]],["Stride", [2,2]],["Padding", [2,2]],["Dilation", [2,2]],["Groups", 1],["Bias", 1],["Padding mode", "reflect"]],"state":{},"interfaces":[["Input",{"id":"ni_16039892157623","value":null}],["Output",{"id":"ni_16039892157624","value":null}]],"position":{"x":312,"y":230},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_16039892316758","name":"MaxPool2d","options":['
  + '["Kernel size", [0,0]],["Stride", [1,1]],["Padding", [1,1]],["Dilation",[1,1]],["Return indices", null],["Ceil mode", null]],'
  + '"state":{},"interfaces":[["Input",{"id":"ni_16039892316759","value":null}],["Output",{"id":"ni_160398923167510","value":null}]],"position":{"x":331,"y":479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160398924063014","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160398924063015","value":null}]],"position":{"x":608,"y":437},"width":200,"twoColumn":false}],"connections":[{"id":"16039892205757","from":"ni_16039892032881","to":"ni_16039892157623"},{"id":"160398923656513","from":"ni_16039892157624","to":"ni_16039892316759"},{"id":"160398961820424","from":"ni_160398923167510","to":"ni_160398924063015"}],"panning":{"x":0,"y":0},"scaling":1}';

const conv2dDefGenerated = 'nn.Conv2d(in_channels=0, out_channels=0, kernel_size=(0,0), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=1, padding_mode=\'zeros\')';
const maxpool2dDefGenerated = 'nn.MaxPool2d(kernel_size=(0,0), stride=(0,0), padding=(0,0))';

describe('model codegen', () => {
  it('renders props.msg when passed', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":31,"y":162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":245,"y":150},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513148090922","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513148090923","value":null}],["Output",{"id":"ni_160513148090924","value":null}]],"position":{"x":454,"y":163},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513148489227","from":"ni_16051304343279","to":"ni_160513148090923"},{"id":"160513148671330","from":"ni_160513148090924","to":"ni_160513044748320"}],"panning":{"x":0,"y":0},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2d_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, input_1):
    x = input_1
    x = self.conv2d_1(x)
    x = self.maxpool2d_1(x)
    return x`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('renders uses var names as modified in the ui', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"named_input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":31,"y":162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2dname","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":245,"y":150},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513148090922","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513148090923","value":null}],["Output",{"id":"ni_160513148090924","value":null}]],"position":{"x":454,"y":163},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513148489227","from":"ni_16051304343279","to":"ni_160513148090923"},{"id":"160513148671330","from":"ni_160513148090924","to":"ni_160513044748320"}],"panning":{"x":0,"y":0},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2dname_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, named_input_1):
    x = named_input_1
    x = self.conv2dname_1(x)
    x = self.maxpool2d_1(x)
    return x`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for concat node', () => {
    const iState = JSON.parse('{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":31,"y":162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16051304343277","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16051304343278","value":null}],["Output",{"id":"ni_16051304343279","value":null}]],"position":{"x":245,"y":150},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":704,"y":147},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513148090922","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513148090923","value":null}],["Output",{"id":"ni_160513148090924","value":null}]],"position":{"x":251,"y":582},"width":200,"twoColumn":false},{"type":"Concat","id":"node_160513194994734","name":"Concat","options":[],"state":{},"interfaces":[["Input 1",{"id":"ni_160513194994835","value":null}],["Input 2",{"id":"ni_160513194994836","value":null}],["Output",{"id":"ni_160513194994837","value":null}]],"position":{"x":493,"y":274},"width":200,"twoColumn":false}],"connections":[{"id":"160513123339511","from":"ni_16051304291466","to":"ni_16051304343278"},{"id":"160513194393533","from":"ni_16051304291466","to":"ni_160513148090923"},{"id":"160513195310840","from":"ni_16051304343279","to":"ni_160513194994835"},{"id":"160513195526843","from":"ni_160513148090924","to":"ni_160513194994836"},{"id":"160513195733046","from":"ni_160513194994837","to":"ni_160513044748320"}],"panning":{"x":14,"y":-145},"scaling":1}');
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2d_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, input_1):
    x = input_1
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
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2d_1 = ${conv2dDefGenerated}
    self.conv2d_2 = ${conv2dDefGenerated}
    self.conv2d_3 = ${conv2dDefGenerated}

  def forward(self, input_1):
    x = input_1
    x = self.conv2d_1(x)
    x_1 = self.conv2d_2(x)
    x_2 = self.conv2d_3(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });

  it('generates code for nested branching', () => {
    const iState = JSON.parse(
      '{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":-366.29515250193157,"y":184.1687612133773},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513044748219","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513044748320","value":null}]],"position":{"x":652,"y":116},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513234026953","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513234026954","value":null}],["Output",{"id":"ni_160513234026955","value":null}]],"position":{"x":210,"y":115},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513234536963","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513234536964","value":null}],["Output",{"id":"ni_160513234536965","value":null}]],"position":{"x":210,"y":275},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160513235923569","name":"output2","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160513235923570","value":null}]],"position":{"x":681,"y":274},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_160513247683377","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_160513247683378","value":null}],["Output",{"id":"ni_160513247683379","value":null}]],"position":{"x":197.98743271973615,"y":497.6585279751281},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513265661692","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513265661693","value":null}],["Output",{"id":"ni_160513265661794","value":null}]],"position":{"x":-133.85565486123528,"y":183.51167021007862},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_1605132697255112","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_1605132697255113","value":null}],["Output",{"id":"ni_1605132697255114","value":null}]],"position":{"x":490.5306997147442,"y":434.7134651802162},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_1605132700587118","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_1605132700587119","value":null}],["Output",{"id":"ni_1605132700587120","value":null}]],"position":{"x":461.585636919831,"y":599.080071765615},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_1605132713760124","name":"output3","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1605132713760125","value":null}]],"position":{"x":720.023697588697,"y":430.57845620951434},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_1605132721065129","name":"output4","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1605132721065130","value":null}]],"position":{"x":691.0786347937841,"y":604.2488329789923},"width":200,"twoColumn":false}],"connections":[{"id":"160513234367862","from":"ni_160513234026955","to":"ni_160513044748320"},{"id":"160513236247273","from":"ni_160513234536965","to":"ni_160513235923570"},{"id":"160513266076297","from":"ni_16051304291466","to":"ni_160513265661693"},{"id":"1605132663333101","from":"ni_160513265661794","to":"ni_160513234026954"},{"id":"1605132665284105","from":"ni_160513265661794","to":"ni_160513234536964"},{"id":"1605132670396111","from":"ni_160513265661794","to":"ni_160513247683378"},{"id":"1605132699289117","from":"ni_160513247683379","to":"ni_1605132697255113"},{"id":"1605132703920123","from":"ni_160513247683379","to":"ni_1605132700587119"},{"id":"1605132716364128","from":"ni_1605132697255114","to":"ni_1605132713760125"},{"id":"1605132723551133","from":"ni_1605132700587120","to":"ni_1605132721065130"}],"panning":{"x":-66.6922802178041,"y":-43.955117448891},"scaling":0.9673497756211787}',
    );
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2d_2 = ${conv2dDefGenerated}
    self.conv2d_3 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}
    self.conv2d_1 = ${conv2dDefGenerated}
    self.conv2d_4 = ${conv2dDefGenerated}
    self.conv2d_5 = ${conv2dDefGenerated}

  def forward(self, input_1):
    x = input_1
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
      '{"nodes":[{"type":"InModel","id":"node_16051304291465","name":"input","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16051304291466","value":null}]],"position":{"x":-360.0926390458788,"y":233.78886886179947},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160513265661692","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160513265661693","value":null}],["Output",{"id":"ni_160513265661794","value":null}]],"position":{"x":-100.77558309562039,"y":140.09407601770917},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_1605133122622134","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_1605133122622135","value":null}],["Output",{"id":"ni_1605133122622136","value":null}]],"position":{"x":-98.70807861027001,"y":319.96696624323977},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_1605133143030140","name":"output","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1605133143030141","value":null}]],"position":{"x":160.76373430127143,"y":166.971634327271},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_1605133152572148","name":"output2","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1605133152572149","value":null}]],"position":{"x":200.04631952293892,"y":349.94578128082804},"width":200,"twoColumn":false}],"connections":[{"id":"160513323127513","from":"ni_16051304291466","to":"ni_160513265661693"},{"id":"160513323127615","from":"ni_16051304291466","to":"ni_1605133122622135"},{"id":"160513323127617","from":"ni_1605133122622136","to":"ni_1605133152572149"},{"id":"160513323127619","from":"ni_160513265661794","to":"ni_1605133143030141"}],"panning":{"x":0,"y":0},"scaling":1}',
    );
    const graph = istateToGraph(iState);
    // TODO: create the graph to test code generation
    let actual = generateModelCode(graph, 'Model');
    actual = removeBlankLines(actual);

    let expected = `
import torch
import torch.nn as nn
import torch.nn.functional as F
# enabling relative imports
import os
import sys
sys.path.insert(0, os.path.join((os.path.abspath(os.path.dirname(sys.argv[0]))), ".."))


class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.conv2d_1 = ${conv2dDefGenerated}
    self.maxpool2d_1 = ${maxpool2dDefGenerated}

  def forward(self, input_1):
    x = input_1
    x_1 = self.conv2d_1(x)
    x_2 = self.maxpool2d_1(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    expect(actual).toBe(expected);
  });
});
