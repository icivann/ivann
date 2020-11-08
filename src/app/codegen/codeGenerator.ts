/* eslint-disable no-param-reassign */
import GraphNode from '@/app/ir/GraphNode';
import { ModelLayerNode } from '@/app/ir/mainNodes';
import InModel from '@/app/ir/InModel';

import OutModel from '@/app/ir/OutModel';
import Graph from '@/app/ir/Graph';
import Concat from '@/app/ir/Concat';
import Custom from '@/app/ir/Custom';

const imports = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F',
].join('\n');

let nodeNames = new Map<GraphNode, string>();
let nodeTypeCounters = new Map<string, number>();

const indent = '  ';

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
  const counter = (nodeTypeCounters.get(nodeType) ?? 0) + 1;
  nodeTypeCounters.set(nodeType, counter);
  const name = `${nodeType}_${counter}`;
  nodeNames.set(node, name);
  return name;
}

function getBranchVar(branchCounter: number): string {
  return (branchCounter === 0) ? 'x' : `x_${branchCounter}`;
}

// returns:
//    code lines generated as array of strings
//    highet branch number it finished on
function generateModelGraphCode(
  node: GraphNode,
  incomingBranch: number,
  branch: number,
  branchesMap: Map<GraphNode, number[]>,
  graph: Graph,
  outputs: Set<string>,
): [string[], number] {
  const nodeName = getNodeName(node);

  let code: string[] = [];

  // TODO: check here for other branching nodes e.g. Concat
  if (node.mlNode instanceof InModel) {
    incomingBranch = incomingBranch !== 0 ? incomingBranch + 1 : incomingBranch;
    code.push(`${getBranchVar(incomingBranch)} = ${nodeName}`);
  } else if (node.mlNode instanceof OutModel) {
    outputs.add(getBranchVar(incomingBranch));
  } else if (node.mlNode instanceof Concat || node.mlNode instanceof Custom) {
    // TODO: test this
    const readyConnections: number[] = branchesMap.get(node) ?? [];
    if (readyConnections.length !== node.inputInterfaces.size - 1) {
      const incomingBranches = branchesMap.get(node);
      if (incomingBranches !== undefined) {
        incomingBranches.push(incomingBranch);
        branchesMap.set(node, incomingBranches);
      } else {
        branchesMap.set(node, [incomingBranch]);
      }
      return [code, incomingBranch];
    }

    readyConnections.push(branch);
    const params = readyConnections.map((branch) => getBranchVar(branch));
    branch += 1;

    code.push(`${getBranchVar(branch)} = ${node.mlNode.callCode(params, '')}`);
  } else {
    code.push(`${getBranchVar(branch)} = self.${nodeName}(${getBranchVar(incomingBranch)})`);
  }

  const children = graph.nextNodesFrom(node);

  // If we only have one child, we pass in our current branch
  if (children.length === 1) {
    const [retCode, retBranch] = generateModelGraphCode(
      children[0],
      branch,
      branch,
      branchesMap,
      graph,
      outputs,
    );

    return [code.concat(retCode), retBranch];
  }

  if (children.length > 1) { // If we have multiple children, we have to split branches
    let childBranch = branch + 1;
    let maxChildBranch = branch;
    children.forEach((child) => {
      const [retCode, retBranch] = generateModelGraphCode(
        child,
        branch,
        childBranch,
        branchesMap,
        graph,
        outputs,
      );

      childBranch = retBranch + 1;
      maxChildBranch = retBranch;

      code = code.concat(retCode);
    });

    return [code, maxChildBranch];
  }

  // If we have no children
  return [[], branch];
}

function generateModel(graph: Graph): string {
  const header = 'class Model(nn.Module):';

  // resetting the naming counters for each new graph
  nodeNames = new Map<GraphNode, string>();
  nodeTypeCounters = new Map<string, number>();
  const outputs: Set<string> = new Set();

  const inputs = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode instanceof InModel);
  const inputNames = inputs.map((node) => getNodeName(node));
  let forward: string[] = [`${indent}def forward(self, ${inputNames.join(', ')}):`];

  // dummy GraphNode to start off the recusrive DFS traversal
  // const startLayer = new InModel([0n]);
  // const startNode = new GraphNode(startLayer, new UUID('startNode'));
  // connections.set(startNode, inputs);
  let currentBranch = 0;
  const branchesMap = new Map<GraphNode, number[]>();

  inputs.forEach((n) => {
    const [retCode, retBranch] = generateModelGraphCode(n, 0, currentBranch, branchesMap, graph,
      outputs);
    currentBranch = retBranch + 1;
    forward = forward.concat(retCode);
  });

  const nodeDefinitions: string[] = [];
  // TODO: sort layer definitions
  graph.nodesAsArray.forEach((n) => {
    if ((n.mlNode as ModelLayerNode).initCode !== undefined) {
      nodeDefinitions.push(`self.${getNodeName(n)} = nn.${(n.mlNode as ModelLayerNode).initCode()}`);
    }
  });
  const init = [`${indent}def __init__(self):`].concat(nodeDefinitions);
  forward.push(`return ${[...outputs].join(', ')}`);
  const forwardMethod = forward.join(`\n${indent}${indent}`);
  const initMethod = init.join(`\n${indent}${indent}`);

  return [header, initMethod, forwardMethod].join('\n\n');
}

function generateFunctions(graph: Graph): string {
  const customNodes = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode instanceof Custom);

  if (customNodes.length === 0) {
    return '';
  }

  const funcs: string[] = [];
  customNodes.forEach((node) => {
    if (node.mlNode instanceof Custom) {
      funcs.push(node.mlNode.code);
    }
  });

  return funcs.join('\n\n');
}

export default function generateCode(graph: Graph): string {
  const funcs = generateFunctions(graph);
  const model = generateModel(graph);

  const result = [imports];

  if (funcs.length > 0) {
    result.push(funcs);
  }

  result.push(model);

  return result.join('\n\n');
}
