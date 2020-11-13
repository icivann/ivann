import GraphNode from '@/app/ir/GraphNode';
import Graph from '@/app/ir/Graph';
import Custom from '@/app/ir/Custom';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';

function generateFunctions(graph: Graph): string {
  const customNodes = graph.nodesAsArray.filter(
    (item: GraphNode) => item.mlNode instanceof Custom,
  );
  // TODO: expand to other training nodes (and custom train)
  const trainNodes = graph.nodesAsArray.filter(
    (item: GraphNode) => item.mlNode instanceof TrainClassifier,
  );

  if (customNodes.length === 0 && trainNodes.length === 0) {
    return '';
  }

  const funcs: string[] = [];
  customNodes.forEach((node) => {
    if (node.mlNode instanceof Custom) {
      funcs.push(node.mlNode.code);
    }
  });

  trainNodes.forEach((node) => {
    if (node.mlNode instanceof TrainClassifier) {
      funcs.push(node.mlNode.initCode());
    }
  });

  return funcs.join('\n\n');
}

export default generateFunctions;
