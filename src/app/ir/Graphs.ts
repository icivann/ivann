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
import Graph from '@/app/ir/Graph';

export default function mnist(): Graph {
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

  const list = [conv, maxPool].map((t) => new GraphNode(t));

  const map = new Map(list.map((t) => [t.uniqueId, t] as [UUID, GraphNode]));
  return new Graph(map, [[list[0].uniqueId, list[1].uniqueId]]);
}
