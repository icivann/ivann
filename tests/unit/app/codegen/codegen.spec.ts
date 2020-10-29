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
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":11,"y":106},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16039996941842","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":262,"y":76},"width":200,"twoColumn":false},{"type":"MaxPooling2D","id":"node_16039998843798","name":"MaxPooling2D","options":[["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"]],"state":{},"interfaces":[["Input",{"id":"ni_16039998843799","value":null}],["Output",{"id":"ni_160399988437910","value":null}]],"position":{"x":552,"y":56},"width":200,"twoColumn":false},{"type":"Operations","id":"node_160399989117214","name":"Concat","options":[],"state":{},"interfaces":[["Input 1",{"id":"ni_160399989117215","value":null}],["Input 2",{"id":"ni_160399989117216","value":null}],["Output",{"id":"ni_160399989117217","value":null}]],"position":{"x":869,"y":195},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160399991199424","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160399991199425","value":null}]],"position":{"x":930,"y":450},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160399992445726","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160399992445727","value":null}],["Output",{"id":"ni_160399992445728","value":null}]],"position":{"x":439,"y":310},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160399988834313","from":"ni_16039996941854","to":"ni_16039998843799"},{"id":"160399989584520","from":"ni_160399988437910","to":"ni_160399989117215"},{"id":"160399992841332","from":"ni_16039996941854","to":"ni_160399992445727"},{"id":"160399993040235","from":"ni_160399992445728","to":"ni_160399989117216"},{"id":"160399994882741","from":"ni_160399989117217","to":"ni_160399991199425"}],"panning":{"x":-499.0027429825419,"y":-0.32002882179401637},"scaling":1.0030022491546675}');
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
    self.conv2d_2 = Conv2D(16, 32, 2,2)
  def forward(self, inmodel_1)
    x = inmodel_1
    x = self.conv2d_1(x)
    x_1 = self.maxpool2d_1(x)
    x_2 = self.conv2d_2(x)
    x_3 = torch.cat(x_1, x_2)
    return x_3`.trim();
    expected = removeBlankLines(expected);

    const diff = diffStringsUnified(expected, actual);
    console.log(diff);
    expect(actual).toMatch(expected);
  });

  it('two outputs', () => {
    const iState = JSON.parse('{"nodes":[{"type":"I/O","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":-77.46078244707978,"y":106},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_16039996941842","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":186.17647218821736,"y":115.49142073530265},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400104763047","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400104763048","value":null}],["Output",{"id":"ni_160400104763149","value":null}]],"position":{"x":495.61690006906053,"y":80.42745956751298},"width":200,"twoColumn":false},{"type":"Convolution2D","id":"node_160400110066953","name":"Convolution2D","options":[["Filters",1],["Kernel Size",[1,1]],["Stride",[1,1]],["Padding","Valid"],["Activation","None"],["Use Bias",1],["Weights Initializer","Xavier"],["Bias Initializer","Zeros"],["Bias Regularizer","None"],["Weights Regularizer","None"]],"state":{},"interfaces":[["Input",{"id":"ni_160400110066954","value":null}],["Output",{"id":"ni_160400110066955","value":null}]],"position":{"x":497.19655689847264,"y":298.4201020263882},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400112448859","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400112448860","value":null}]],"position":{"x":911.0666462044529,"y":29.87844102632479},"width":200,"twoColumn":false},{"type":"I/O","id":"node_160400113171465","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400113171466","value":null}]],"position":{"x":939.5004691338713,"y":306.3183861734489},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160400105094252","from":"ni_16039996941854","to":"ni_160400104763048"},{"id":"160400110318058","from":"ni_16039996941854","to":"ni_160400110066954"},{"id":"160400112847964","from":"ni_160400104763149","to":"ni_160400112448860"},{"id":"160400114003271","from":"ni_160400110066955","to":"ni_160400113171466"}],"panning":{"x":115.71029291343712,"y":41.20611629722157},"scaling":0.6330488884551875}');
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
});
