import Adadelta from '@/app/ir/overview/optimizers/Adadelta';
import Model from '@/app/ir/model/model';
import GraphNode from '@/app/ir/GraphNode';
import Graph from '@/app/ir/Graph';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';

function generateTrainingPipeline(node: GraphNode, graph: Graph): string[] {
  // traverse each incoming connection backwards and link up

  const trainNode = node.mlNode as TrainClassifier;

  // const incomingNodes = graph.prevNodesFrom(node);
  const optimizer = graph.nodesAsArray.filter(
    (item: GraphNode) => item.mlNode instanceof Adadelta,
  )[0].mlNode as Adadelta;

  const model = graph.nodesAsArray.filter(
    (item: GraphNode) => item.mlNode instanceof Model,
  )[0].mlNode as Model;

  const body = [];

  const device = `"${trainNode.Device}"`;
  body.push(`device = ${device}`);
  const modelName = `${model.name}_model`;
  body.push(`${modelName} = ${model.name}().to(device)`);
  body.push(`optimizer = ${optimizer.initCode(`${modelName}.parameters()`)}`);

  // (model, train_loader, test_loader, optimizer, device, epoch
  body.push(trainNode.callCode([
    modelName,
    'None',
    'None',
    'optimizer',
    device,
    trainNode.Epochs.toString(),
  ]));

  return body;
}

export default generateTrainingPipeline;
