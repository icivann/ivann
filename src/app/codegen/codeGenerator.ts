/* eslint-disable no-param-reassign */
import GraphNode from '@/app/ir/GraphNode';
import { ModelLayerNode, OverviewCallableNode, OverviewNode } from '@/app/ir/mainNodes';
import InModel from '@/app/ir/InModel';

import OutModel from '@/app/ir/OutModel';
import Graph from '@/app/ir/Graph';
import Concat from '@/app/ir/Concat';
import Flatten from '@/app/ir/model/flatten';
import Custom from '@/app/ir/Custom';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';
import Adadelta from '@/app/ir/overview/optimizers/adadelta';
import Model from '@/app/ir/model/model';

import { indent, getNodeType } from '@/app/codegen/common';

import generateData from '@/app/codegen/dataGenerator';
import OverviewCustom from '@/app/ir/overview/OverviewCustom';

const imports = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F',
  'from torch.utils.data import Dataset, DataLoader',
  'from torchvision import transforms',
  'import pandas as pd',
  'import numpy as np',
  'import os',
  'from PIL import Image',
].join('\n');

function getNodeName(
  node: GraphNode,
  nodeNames: Map<GraphNode, string>,
  nodeTypeCounters: Map<string, number>,
): string {
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
  nodeNames: Map<GraphNode, string>,
  nodeTypeCounters: Map<string, number>,
): [string[], number] {
  const nodeName = getNodeName(node, nodeNames, nodeTypeCounters);

  let code: string[] = [];

  // TODO: check here for other branching nodes e.g. Concat
  if (node.mlNode instanceof InModel) {
    incomingBranch = incomingBranch !== 0 ? incomingBranch + 1 : incomingBranch;
    code.push(`${getBranchVar(incomingBranch)} = ${nodeName}`);
  } else if (node.mlNode instanceof OutModel) {
    outputs.add(getBranchVar(incomingBranch));
  } else if (node.mlNode instanceof Concat
    || node.mlNode instanceof Flatten
    || node.mlNode instanceof Custom) {
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
      nodeNames,
      nodeTypeCounters,
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
        nodeNames,
        nodeTypeCounters,
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

function generateModel(graph: Graph, name: string): string {
  const header = `class ${name}(nn.Module):`;

  // resetting the naming counters for each new graph
  const nodeNames = new Map<GraphNode, string>();
  const nodeTypeCounters = new Map<string, number>();
  const outputs: Set<string> = new Set();

  const inputs = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode instanceof InModel);
  const inputNames = inputs.map((node) => getNodeName(node, nodeNames, nodeTypeCounters));
  let forward: string[] = [`${indent}def forward(self, ${inputNames.join(', ')}):`];

  // dummy GraphNode to start off the recusrive DFS traversal
  // const startLayer = new InModel([0n]);
  // const startNode = new GraphNode(startLayer, new UUID('startNode'));
  // connections.set(startNode, inputs);
  let currentBranch = 0;
  const branchesMap = new Map<GraphNode, number[]>();

  inputs.forEach((n) => {
    const [retCode, retBranch] = generateModelGraphCode(n, 0, currentBranch, branchesMap, graph,
      outputs, nodeNames, nodeTypeCounters);
    currentBranch = retBranch + 1;
    forward = forward.concat(retCode);
  });

  const nodeDefinitions: string[] = [];
  // TODO: sort layer definitions
  graph.nodesAsArray.forEach((n) => {
    if ((n.mlNode as ModelLayerNode).initCode !== undefined) {
      nodeDefinitions.push(`self.${getNodeName(n, nodeNames, nodeTypeCounters)} = nn.${(n.mlNode as ModelLayerNode).initCode()}`);
    }
  });
  const init = [`${indent}def __init__(self):`, `super(${name}, self).__init__()`].concat(nodeDefinitions);
  forward.push(`return ${[...outputs].join(', ')}`);
  const forwardMethod = forward.join(`\n${indent}${indent}`);
  const initMethod = init.join(`\n${indent}${indent}`);

  return [header, initMethod, forwardMethod].join('\n\n');
}

function generateFunctions(graph: Graph): string {
  const customNodes = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode instanceof Custom);
  // TODO: expand to other training nodes (and custom train)

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

export function generateModelCode(graph: Graph, name: string): string {
  const funcs = generateFunctions(graph);
  const model = generateModel(graph, name);

  const result = [];

  if (funcs.length > 0) {
    result.push(funcs);
  }

  result.push(model);

  return result.join('\n\n');
}

function isNodeTrainer(node: GraphNode): boolean {
  return node.mlNode instanceof TrainClassifier
    || (node.mlNode instanceof OverviewCustom && node.mlNode.trainer);
}

function generateOverviewGraphCode(
  node: GraphNode,
  graph: Graph,
  nodeNames: Map<GraphNode, string>,
  nodeTypeCounters: Map<string, number>,
): [string[], string] {
  const prevNodes = graph.prevNodesFrom(node);
  let code: string[] = [];
  const params: string[] = [];

  prevNodes.forEach((prevNode) => {
    const [prevCode, prevName] = generateOverviewGraphCode(prevNode, graph, nodeNames,
      nodeTypeCounters);
    params.push(prevName);
    code = code.concat(prevCode);
  });

  const isNewNode = !nodeNames.has(node);
  const name = getNodeName(node, nodeNames, nodeTypeCounters);
  if (isNodeTrainer(node)) {
    console.log('heyyo there');
    code = code.concat(`${(node.mlNode as OverviewCallableNode).callCode(params)}`);
  } else if (isNewNode && (node.mlNode as OverviewNode).initCode !== undefined) {
    code = code.concat(`${name} = ${(node.mlNode as OverviewNode).initCode(params)[0]}`);
  }
  return [code, name];
}

function generateTrainingPipeline(node: GraphNode, graph: Graph): string[] {
  // traverse each incoming connection backwards and link up
  let body: string[] = [];
  const nodeNames = new Map<GraphNode, string>();
  const nodeTypeCounters = new Map<string, number>();
  const [nodeCode, nodeName] = generateOverviewGraphCode(node, graph, nodeNames, nodeTypeCounters);
  body = body.concat(nodeCode);
  return body;
}

function generateOverview(graph: Graph): string {
  let main = ['def main():'];

  const trainNodes = graph.nodesAsArray.filter(isNodeTrainer);

  let funcs: string[] = [];

  trainNodes.forEach((node) => {
    funcs = funcs.concat((node.mlNode as OverviewCallableNode).initCode());
  });

  // Create functions for training

  // const main = [header, body.join(`\n${indent}`)];
  const entry = `if __name__ == '__main__':\n${indent}main()`;

  if (trainNodes.length === 0) {
    main.push(`${indent}${indent}pass`);
  } else {
    trainNodes.forEach((node) => {
      main = main.concat(generateTrainingPipeline(node, graph));
    });
  }

  return [funcs, main.join(`\n${indent}`), entry].join('\n\n');
}

export function generateOverviewCode(
  graph: Graph,
  modelEditors: [Graph, string][],
  dataEditors: [Graph, string][],
): string {
  // TODO: beware of duplicate custom functions
  const funcs = generateFunctions(graph);

  const models = modelEditors.map((editor) => generateModelCode(editor[0], editor[1])).join('\n\n');

  const overview = generateOverview(graph);

  const datasets = dataEditors.map((editor) => generateData(editor[0], editor[1])).join('\n\n');

  const result = [imports];

  if (funcs.length > 0) {
    result.push(funcs);
  }

  result.push(datasets);
  result.push(models);

  result.push(overview);

  return result.join('\n\n');
}

export default generateOverviewCode;
