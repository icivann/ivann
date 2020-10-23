import Graph from '@/app/ir/Graph';
import GraphNode from '@/app/ir/GraphNode';
import Conv2D from '@/app/ir/conv/Conv2D';
import { UUID } from '@/app/util';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Initializer,
  Padding,
  Regularizer,
} from '@/app/ir/irCommon';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import { Stack } from 'stack-typescript';
import { MlNode, ModelNode } from '@/app/ir/mainNodes';
import InModel from '../ir/InModel';

import OutModel from '../ir/OutModel';

const imports = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F'].join('\n');

const nodeNames = new Map<GraphNode, string>();
const nodeTypeCounters = new Map<string, bigint>();

function getNodeType(node: GraphNode): string {
  return node.mlNode.constructor.name.toLowerCase();
}

function getNodeName(node: GraphNode): string {
  if (nodeNames.has(node)) {
    // TODO: linting complains about nodeNames.get possibly being undefined
    // when I've already checked with nodeNames.has
    return nodeNames.get(node) ?? '';
  }
  const nodeType = getNodeType(node);
  const counter = (nodeTypeCounters.get(nodeType) ?? 0n) + 1n;
  nodeTypeCounters.set(nodeType, counter);
  const name = `${nodeType}_${counter}`;
  nodeNames.set(node, name);
  return name;
}

function generateModel(nodes: Set<GraphNode>, connections: Map<GraphNode, [GraphNode]>): string {
  const result: string[] = [];
  const init: string[] = [];

  const inputs = [...nodes.values()].filter((item: GraphNode) => item.mlNode instanceof InModel);
  const inputNames = inputs.map((node) => getNodeName(node));
  const forward: string[] = [`\tdef forward(${inputNames.join(', ')})`];

  console.log(nodes);
  const stack = new Stack<GraphNode>(...inputs);

  let branchCounter = 0;
  while (stack.length > 0) {
    const node = stack.pop();
    const nodeName = getNodeName(node);
    // TODO: check here for other branching nodes
    if (node.mlNode instanceof InModel) {
      branchCounter += 1;
      forward.push(`x_${branchCounter} = ${nodeName}`);
    } else {
      forward.push(`x_${branchCounter} = self.${nodeName}(x_${branchCounter})`);
      const nodeConnections = connections.get(node);
      // TODO: add outgoing connections onto stack etc
    }
  }

  const header = 'class Mode(nn.Module):';
  const forwardMethod = forward.join('\n\t\t');

  return [header, forwardMethod].join('\n');
}

export default function generateCode(): string {
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
  const maxPool = new MaxPool2D(Padding.Same, [28n, 28n], [2n, 2n]);

  const list = [input, conv, maxPool, output].map((t) => new GraphNode(t));

  const map = new Map(list.map((t) => [t.uniqueId, t] as [UUID, GraphNode]));

  const nodes = new Set<GraphNode>(list);
  const connections = new Map<GraphNode, [GraphNode]>([
    [list[0], [list[1]]],
    [list[1], [list[2]]],
    [list[2], [list[3]]],
    [list[3], [list[4]]],
  ]);

  const model = generateModel(nodes, connections);
  const result = [imports, model].join('\n');
  return result;
}

// function modelInit(nodes: Set<ModelNode>) {
// }
