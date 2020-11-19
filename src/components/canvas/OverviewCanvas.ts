import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import Model from '@/nodes/overview/Model';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import TrainClassifier from '@/nodes/overview/train/TrainClassifier';
import Adadelta from '@/nodes/overview/optimizers/Adadelta';
import { Data } from '@/nodes/overview/Data';

export default class OverviewCanvas extends AbstractCanvas {
  nodeList = [
    {
      category: OverviewCategories.Model,
      nodes: [
        {
          name: OverviewNodes.ModelNode,
          node: Model,
        },
      ],
    },
    {
      category: OverviewCategories.Data,
      nodes: [
        {
          name: OverviewNodes.DataNode,
          node: Data,
        },
      ],
    },
    {
      category: OverviewCategories.Train,
      nodes: [
        {
          name: OverviewNodes.TrainClassifier,
          node: TrainClassifier,
        },
      ],
    },
    {
      category: OverviewCategories.Optimizer,
      nodes: [
        {
          name: OverviewNodes.Adadelta,
          node: Adadelta,
        },
      ],
    },
  ];
}
