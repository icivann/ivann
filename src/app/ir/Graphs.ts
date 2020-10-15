import GraphNode from '@/app/ir/GraphNode';
import Conv2D from '@/app/ir/Conv2D';
import { UUID } from '@/app/util';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Initializer,
  Padding,
  Regularizer,
} from '@/app/ir/irCommon';
import MaxPool2D from '@/app/ir/maxPool2D';

export default function mnist(): GraphNode[] {
  const zs = BuiltinInitializer.Zeroes;
  const none = BuiltinRegularizer.None;
  const defaultWeights: [Initializer, Regularizer] = [zs, none];
  const conv = new Conv2D(
    new Set(),
    32n,
    Padding.Same,
    defaultWeights,
    null,
    new UUID('theInNode'),
    BuiltinActivationF.Relu,
    [28n, 28n],
    [2n, 2n],
  );
  const convGraph = new GraphNode(conv);

  const maxPool = new MaxPool2D(
    new Set(), convGraph.uniqueId, Padding.Same, [28n, 28n], [2n, 2n],
  );
  const maxPoolGraph = new GraphNode(maxPool);
  conv.outputs.add(maxPoolGraph.uniqueId);

  return [convGraph, maxPoolGraph];
}
