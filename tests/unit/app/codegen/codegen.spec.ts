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

function removeBlankLines(x: string): string {
  return x.trim();
}

describe('codegen', () => {
  it('renders props.msg when passed', () => {
    const input = new InModel([256n, 256n, 3n]);
    const output = new OutModel();

    const zs = BuiltinInitializer.Zeroes;
    const none = BuiltinRegularizer.None;
    const defaultWeights: [Initializer, Regularizer] = [zs, none];
    const conv = new Conv2D(
      32n,
      Padding.Same,
      defaultWeights,
      null,
      BuiltinActivationF.Relu,
      [28n, 28n],
      [2n, 2n],
    );
    const maxPool = new MaxPool2D(Padding.Same, [3n, 3n], [2n, 2n]);

    const nodeConnections = [[input, conv], [conv, maxPool], [maxPool, output]];
    const list = [input, conv, maxPool, output].map((t) => new GraphNode(t));

    const nodes = new Set<GraphNode>(list);

    // TODO: create the graph to test code generation
    let actual = generateCode(new Graph(nodes, []));
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

    // const diff = diffStringsUnified(expected, actual);
    // console.log(diff);
    expect(actual).toMatch(expected);
  });
});
