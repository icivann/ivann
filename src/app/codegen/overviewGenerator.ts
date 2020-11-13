/* eslint-disable no-param-reassign */
import GraphNode from '@/app/ir/GraphNode';
import { ModelLayerNode, TrainNode } from '@/app/ir/mainNodes';
import InModel from '@/app/ir/InModel';

import OutModel from '@/app/ir/OutModel';
import Graph from '@/app/ir/Graph';
import Concat from '@/app/ir/Concat';
import Custom from '@/app/ir/Custom';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';

import { indent, getNodeType } from '@/app/codegen/common';

import generateData from '@/app/codegen/dataGenerator';
import generateFunctions from '@/app/codegen/customGenerator';
import generateModel from '@/app/codegen/modelGenerator';
import generateTrainingPipeline from '@/app/codegen/trainGenerator';

const imports = [
  'import torch',
  'import torch.nn as nn',
  'import torch.nn.functional as F',
  'from torch.utils.data import Dataset, DataLoader',
  'from torchvision import transforms',
].join('\n');

export function generateModelCode(graph: Graph, name: string): string {
  const funcs = generateFunctions(graph);
  const model = generateModel(graph, name);

  const result = [imports];

  if (funcs.length > 0) {
    result.push(funcs);
  }

  result.push(model);

  return result.join('\n\n');
}

function generateOverview(graph: Graph): string {
  let main = ['def main():'];

  const trainNodes = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode
  instanceof TrainClassifier);

  trainNodes.forEach((node) => {
    main = main.concat(generateTrainingPipeline(node, graph));
  });

  // const main = [header, body.join(`\n${indent}`)];
  const entry = `if __name__ == '__main__':\n${indent}main()`;

  return [main.join(`\n${indent}`), entry].join('\n\n');
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

  const data = dataEditors.map((editor) => generateData(editor[0], editor[1])).join('\n\n');
  // console.log(overview);

  const result = [imports, data];

  if (funcs.length > 0) {
    result.push(funcs);
  }

  result.push(models);

  result.push(overview);

  return result.join('\n\n');
}

export default generateOverviewCode;
