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
});
