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
import { ModelLayerNode } from '@/app/ir/mainNodes';
import InModel from '../ir/InModel';

import OutModel from '../ir/OutModel';
import Graph from '../ir/Graph';
import Concat from '../ir/Concat';

const imports = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F'].join('\n');

let nodeNames = new Map<GraphNode, string>();
let nodeTypeCounters = new Map<string, number>();

const indent = '  ';

function getNodeType(node: GraphNode): string {
  return node.modelNode.constructor.name.toLowerCase();
}

function getNodeName(node: GraphNode): string {
  if (nodeNames.has(node)) {
    // TODO: linting complains about nodeNames.get possibly being undefined
    // when I've already checked with nodeNames.has
    return nodeNames.get(node) ?? '';
  }
  const nodeType = getNodeType(node);
  const counter = (nodeTypeCounters.get(nodeType) ?? 0) + 1;
  nodeTypeCounters.set(nodeType, counter);
  const name = `${nodeType}_${counter}`;
  nodeNames.set(node, name);
  return name;
}

function getBrachVar(branchCounter: number): string {
  return (branchCounter === 0) ? 'x' : `x_${branchCounter}`;
}

// returns:
//    code lines generated as array of strings
//    highet branch number it finished on
function generateModelGraphCode(node: GraphNode, incomingBranch: number, branchesMap:
  Map<GraphNode, number[]>, graph: Graph, outputs: Set<string>):
  [string[], number] {
  let currentBranch = incomingBranch;
  const nodeName = getNodeName(node);

  let code: string[] = [];

  // TODO: check here for other branching nodes e.g. Concat
  if (node.modelNode instanceof InModel) {
    if (currentBranch !== 0) { currentBranch += 1; }
    code.push(`${getBrachVar(currentBranch)} = ${nodeName}`);
  } else if (node.modelNode instanceof OutModel) {
    outputs.add(getBrachVar(currentBranch));
  } else if (node.modelNode instanceof Concat) {
    // TODO: test this
    const readyConnections: number[] = branchesMap.get(node) ?? [];
    if (readyConnections.length !== node.inputInterfaces.size) {
      const incomingBranches = branchesMap.get(node);
      if (incomingBranches !== undefined) {
        incomingBranches.push(currentBranch);
        branchesMap.set(node, incomingBranches);
      }
      return [code, currentBranch];
    }
    const params = readyConnections.map((branch) => getBrachVar(branch)).join(',');
    code.push(`${getBrachVar(currentBranch)} = self.${nodeName}(${params})`);
  } else {
    code.push(`${getBrachVar(currentBranch)} = self.${nodeName}(${getBrachVar(currentBranch)})`);
  }

  graph.nextNodesFrom(node).forEach((n) => {
    const incomingBranches = branchesMap.get(n);
    if (incomingBranches !== undefined) {
      incomingBranches.push(1);
      branchesMap.set(n, incomingBranches);
    } else {
      branchesMap.set(n, [currentBranch]);
    }
    const [retCode, retBranch] = generateModelGraphCode(n, currentBranch, branchesMap, graph,
      outputs);
    currentBranch = Math.max(currentBranch, retBranch);
    code = code.concat(retCode);
  });

  return [code, currentBranch];
}

function generateModel(graph: Graph): string {
  const header = 'class Model(nn.Module):';

  // resetting the naming counters for each new graph
  nodeNames = new Map<GraphNode, string>();
  nodeTypeCounters = new Map<string, number>();
  const outputs: Set<string> = new Set();

  const inputs = graph.nodesAsArray.filter((item: GraphNode) => item.modelNode instanceof InModel);
  const inputNames = inputs.map((node) => getNodeName(node));
  let forward: string[] = [`${indent}def forward(self, ${inputNames.join(', ')})`];

  // dummy GraphNode to start off the recusrive DFS traversal
  // const startLayer = new InModel([0n]);
  // const startNode = new GraphNode(startLayer, new UUID('startNode'));
  // connections.set(startNode, inputs);
  let currentBranch = 0;
  const branchesMap = new Map<GraphNode, number[]>();

  inputs.forEach((n) => {
    const [retCode, retBranch] = generateModelGraphCode(n, currentBranch, branchesMap, graph,
      outputs);
    currentBranch = Math.max(currentBranch, retBranch);
    forward = forward.concat(retCode);
  });

  const nodeDefinitions: string[] = [];
  graph.nodesAsArray.forEach((n) => {
    if ((n.modelNode as ModelLayerNode).initCode !== undefined) {
      nodeDefinitions.push(`self.${getNodeName(n)} = ${(n.modelNode as ModelLayerNode).initCode()}`);
    }
  });
  const init = [`${indent}def __init__(self):`].concat(nodeDefinitions);
  forward.push(`return ${[...outputs].join(', ')}`);
  const forwardMethod = forward.join(`\n${indent}${indent}`);
  const initMethod = init.join(`\n${indent}${indent}`);

  return [header, initMethod, forwardMethod].join('\n');
}

export default function generateCode(graph: Graph): string {
  const model = generateModel(graph);
  const result = [imports, model].join('\n\n');
  return result;
}
