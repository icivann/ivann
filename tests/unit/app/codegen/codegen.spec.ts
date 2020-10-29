import GraphNode from '@/app/ir/GraphNode';
import Conv2D from '@/app/ir/conv/Conv2D';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Initializer,
  Padding,
  Regularizer,
} from '@/app/ir/irCommon';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';
import generateCode from '@/app/codegen/codeGenerator';
import Graph from '@/app/ir/Graph';
import istateToGraph from '@/app/ir/istateToGraph';
import { diffStringsUnified } from 'jest-diff';

function removeBlankLines(x: string): string {
  return x.trim();
}

describe('codegen', () => {
  it('renders props.msg when passed', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16039892032880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039892032881","value":null}]],"position":{"x":21,"y":234},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16039892157612","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039892157623","value":null}],["Output",{"id":"ni_16039892157624","value":null}]],"position":{"x":312,"y":230},"width":200,"twoColumn":false},{"type":"MaxPooling2D","id":"node_16039892316758","name":"MaxPooling2D","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_16039892316759","value":null}],["Output",{"id":"ni_160398923167510","value":null}]],"position":{"x":331,"y":479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160398924063014","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160398924063015","value":null}]],"position":{"x":608,"y":437},"width":200,"twoColumn":false}],"connections":[{"id":"16039892205757","from":"ni_16039892032881","to":"ni_16039892157623"},{"id":"160398923656513","from":"ni_16039892157624","to":"ni_16039892316759"},{"id":"160398961820424","from":"ni_160398923167510","to":"ni_160398924063015"}],"panning":{"x":0,"y":0},"scaling":1}');
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
    self.conv2d_1 = Conv2D(16, 32, 2,2)
    self.maxpool2d_1 = MaxPool2d((28,28))
  def forward(self, inmodel_1)
    x = inmodel_1
    x = self.conv2d_1(x)
    x = self.maxpool2d_1(x)
    return x`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });
  it('concat branch test', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16040096081900","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040096081911","value":null}]],"position":{"x":8.773945375544749,"y":169.22570492635924},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400969358118","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400969358219","value":null}]],"position":{"x":1005.9686556627672,"y":158},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400986337934","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400986337935","value":null}],["Output",{"id":"ni_160400986337936","value":null}]],"position":{"x":341.44307155028986,"y":105.230719863825},"width":200,"twoColumn":false},{"type":"MaxPooling2D","id":"node_160400987431140","name":"MaxPooling2D","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_160400987431141","value":null}],["Output",{"id":"ni_160400987431142","value":null}]],"position":{"x":351.3786343853439,"y":315.29690551925114},"width":200,"twoColumn":false},{"type":"Operations","id":"node_160400988154146","name":"Concat","options":[],"state":{},"interfaces":[["Input 1",{"id":"ni_160400988154147","value":null}],["Input 2",{"id":"ni_160400988154148","value":null}],["Output",{"id":"ni_160400988154149","value":null}]],"position":{"x":692.0265030157651,"y":171.940927470616},"width":200,"twoColumn":false}],"connections":[{"id":"160400986848639","from":"ni_16040096081911","to":"ni_160400986337935"},{"id":"160400987670845","from":"ni_16040096081911","to":"ni_160400987431141"},{"id":"160400989758452","from":"ni_160400986337936","to":"ni_160400988154147"},{"id":"160400990025955","from":"ni_160400987431142","to":"ni_160400988154148"},{"id":"160400992456061","from":"ni_160400988154149","to":"ni_160400969358219"}],"panning":{"x":74.43120140268195,"y":165.8682089212183},"scaling":0.7045398550853204}');
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
    self.conv2d_1 = Conv2D(16, 32, 2,2)
    self.maxpool2d_1 = MaxPool2d((28,28))
  def forward(self, inmodel_1)
    x = inmodel_1
    x_1 = self.conv2d_1(x)
    x_2 = self.maxpool2d_1(x)
    x_3 = torch.cat(x_1, x_2)
    return x_3`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });

  it('two outputs', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":-77.46078244707978,"y":106},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16039996941842","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":186.17647218821736,"y":115.49142073530265},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400104763047","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400104763048","value":null}],["Output",{"id":"ni_160400104763149","value":null}]],"position":{"x":503.7178227039244,"y":59.36506071686672},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400110066953","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400110066954","value":null}],["Output",{"id":"ni_160400110066955","value":null}]],"position":{"x":508.5378485872821,"y":288.9421610499153},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400112448859","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400112448860","value":null}]],"position":{"x":911.0666462044529,"y":29.87844102632479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400113171465","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400113171466","value":null}]],"position":{"x":939.5004691338713,"y":306.3183861734489},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160400105094252","from":"ni_16039996941854","to":"ni_160400104763048"},{"id":"160400110318058","from":"ni_16039996941854","to":"ni_160400110066954"},{"id":"160400112847964","from":"ni_160400104763149","to":"ni_160400112448860"},{"id":"160400641665775","from":"ni_160400110066955","to":"ni_160400113171466"}],"panning":{"x":177.3940230694035,"y":189.50554993466366},"scaling":0.6172136465643434}');
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
    self.conv2d_1 = Conv2D(16, 32, 2,2)
    self.conv2d_2 = Conv2D(16, 32, 2,2)
    self.conv2d_3 = Conv2D(16, 32, 2,2)
  def forward(self, inmodel_1)
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = self.conv2d_2(x)
    x_2 = self.conv2d_3(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });
  it('nested branching', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":-101.39562776264694,"y":133.61712921026987},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16039996941842","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":178.07554955335334,"y":141.4143731668673},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400104763047","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400104763048","value":null}],["Output",{"id":"ni_160400104763149","value":null}]],"position":{"x":503.7178227039244,"y":59.36506071686672},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400110066953","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400110066954","value":null}],["Output",{"id":"ni_160400110066955","value":null}]],"position":{"x":508.5378485872821,"y":288.9421610499153},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400112448859","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400112448860","value":null}]],"position":{"x":911.0666462044529,"y":29.87844102632479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400113171465","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400113171466","value":null}]],"position":{"x":939.5004691338713,"y":306.3183861734489},"width":200,"twoColumn":false},{"type":"MaxPooling2D","id":"node_160400907253476","name":"MaxPooling2D","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_160400907253477","value":null}],["Output",{"id":"ni_160400907253578","value":null}]],"position":{"x":467.03535191811665,"y":543.8685871785077},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400908547183","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400908547284","value":null}],["Output",{"id":"ni_160400908547285","value":null}]],"position":{"x":861.0397286512999,"y":536.5040193891028},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400909098789","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400909098890","value":null}],["Output",{"id":"ni_160400909098891","value":null}]],"position":{"x":849.9928769671922,"y":751.9176272292082},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400911038095","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400911038196","value":null}]],"position":{"x":1232.4724805786134,"y":514.4538965152768},"width":200,"twoColumn":false},{"type":"I/O","id":"node_1604009132060100","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_1604009132060101","value":null}]],"position":{"x":1228.1276071453376,"y":707.8007642960396},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160400105094252","from":"ni_16039996941854","to":"ni_160400104763048"},{"id":"160400110318058","from":"ni_16039996941854","to":"ni_160400110066954"},{"id":"160400112847964","from":"ni_160400104763149","to":"ni_160400112448860"},{"id":"160400641665775","from":"ni_160400110066955","to":"ni_160400113171466"},{"id":"160400908003482","from":"ni_16039996941854","to":"ni_160400907253477"},{"id":"160400908905288","from":"ni_160400907253578","to":"ni_160400908547284"},{"id":"160400909363194","from":"ni_160400907253578","to":"ni_160400909098890"},{"id":"160400911382599","from":"ni_160400908547285","to":"ni_160400911038196"},{"id":"1604009155725107","from":"ni_160400909098891","to":"ni_1604009132060101"}],"panning":{"x":372.9582530167131,"y":321.9342393902725},"scaling":0.46031260305141036}');
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
    self.conv2d_1 = Conv2D(16, 32, 2,2)
    self.conv2d_2 = Conv2D(16, 32, 2,2)
    self.conv2d_3 = Conv2D(16, 32, 2,2)
    self.maxpool2d_1 = MaxPool2d((28,28))
    self.conv2d_4 = Conv2D(16, 32, 2,2)
    self.conv2d_5 = Conv2D(16, 32, 2,2)
  def forward(self, inmodel_1)
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = self.conv2d_2(x)
    x_2 = self.conv2d_3(x)
    x_3 = self.maxpool2d_1(x)
    x_4 = self.conv2d_4(x_3)
    x_5 = self.conv2d_5(x_3)
    return x_1, x_2, x_4, x_5`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });
  it('branching input', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16040096081900","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040096081911","value":null}]],"position":{"x":40,"y":182},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16040096333102","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16040096333103","value":null}],["Output",{"id":"ni_16040096333104","value":null}]],"position":{"x":359,"y":152},"width":200,"twoColumn":false},{"type":"MaxPooling2D","id":"node_160400966555215","name":"MaxPooling2D","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_160400966555216","value":null}],["Output",{"id":"ni_160400966555217","value":null}]],"position":{"x":375,"y":335},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400969358118","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400969358219","value":null}]],"position":{"x":891,"y":158},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400969919323","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400969919324","value":null}]],"position":{"x":868,"y":318},"width":200,"twoColumn":false}],"connections":[{"id":"16040096366997","from":"ni_16040096081911","to":"ni_16040096333103"},{"id":"160400969689622","from":"ni_16040096333104","to":"ni_160400969358219"},{"id":"160400970457030","from":"ni_16040096081911","to":"ni_160400966555216"},{"id":"160400971942533","from":"ni_160400966555217","to":"ni_160400969919324"}],"panning":{"x":0,"y":0},"scaling":1}');
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
    self.conv2d_1 = Conv2D(16, 32, 2,2)
    self.maxpool2d_1 = MaxPool2d((28,28))
  def forward(self, inmodel_1)
    x = inmodel_1
    x_1 = self.conv2d_1(x)
    x_2 = self.maxpool2d_1(x)
    return x_1, x_2`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });
});
