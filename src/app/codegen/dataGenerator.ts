import GraphNode from '@/app/ir/GraphNode';
import Graph from '@/app/ir/Graph';
import InData from '@/app/ir/data/InData';
import ToTensor from '@/app/ir/data/ToTensor';
import Grayscale from '@/app/ir/data/Grayscale';

import { indent, getNodeType } from '@/app/codegen/common';

const nodeNames = new Map<GraphNode, string>();
const nodeTypeCounters = new Map<string, number>();

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

function generateDataTransformCode(
  node: GraphNode,
  incomingBranch: number,
  branch: number,
  branchesMap: Map<GraphNode, number[]>,
  graph: Graph,
  outputs: Set<string>,
): [string[], number] {
  let code: string[] = [];

  // TODO do in a systematic way
  if (node.mlNode instanceof ToTensor || node.mlNode instanceof Grayscale) {
    code.push(node.mlNode.initCode());
  }

  const children = graph.nextNodesFrom(node);

  // If we only have one child, we pass in our current branch
  if (children.length === 1) {
    const [retCode, retBranch] = generateDataTransformCode(
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
      const [retCode, retBranch] = generateDataTransformCode(
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

  return [[], branch];
}

function generateData(graph: Graph, dataName: string): string {
  const header = `class ${dataName}(Dataset):`;

  const inputs = graph.nodesAsArray.filter((item: GraphNode) => item.mlNode instanceof InData);
  const inputNames = inputs.map((node) => getNodeName(node));
  const init: string[] = [`${indent}def __init__(self, ${inputNames.join(', ')}):`];

  inputNames.forEach((name) => {
    init.push(`${indent}${indent}self.${name} = ${name}`);
  });

  const len: string[] = [`${indent}def __len__(self):`];
  len.push(`${indent}${indent}return len(self.${inputNames[0]})`);

  const getitem: string[] = [`${indent}def __getitem__(self, idx):`];

  const transforms: string[][] = [];

  let currentBranch = 0;
  const branchesMap = new Map<GraphNode, number[]>();
  const outputs: Set<string> = new Set();
  inputs.forEach((input) => {
    const [retCode, retBranch] = generateDataTransformCode(
      input,
      0,
      currentBranch,
      branchesMap,
      graph,
      outputs,
    );

    currentBranch = retBranch + 1;
    transforms.push(retCode);
  });

  const transformsInitCode: string[] = [];
  const transformsCallCode: string[] = [];
  for (let i = 0; i < inputs.length; i += 1) {
    const input = inputNames[i];
    const transform = transforms[i];

    getitem.push(`${indent}${indent}${input} = self.${input}[idx]`);

    if (transform.length > 0) {
      transformsInitCode.push(`${indent}${indent}self.transform_${input} = transforms.Compose([\n${indent}${indent}${indent}${transform.join(`,\n${indent}${indent}${indent}`)},\n${indent}${indent}])`);
      transformsCallCode.push(`${indent}${indent}${input} = self.transform_${input}(${input})`);
    }
  }

  transformsInitCode.forEach((c) => {
    init.push(c);
  });

  transformsCallCode.forEach((c) => {
    getitem.push(c);
  });

  getitem.push(`${indent}${indent}return ${inputNames.join(', ')}`);

  return [
    header,
    init.join('\n'),
    len.join('\n'),
    getitem.join('\n'),
  ].join('\n\n');
}

export default generateData;
