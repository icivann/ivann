import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import Model from '@/nodes/overview/Model';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import TrainClassifier from '@/nodes/overview/train/TrainClassifier';
import Adadelta from '@/nodes/overview/optimizers/Adadelta';
import { Data } from '@/nodes/overview/Data';
import { Editor } from '@baklavajs/core';
import OverviewCustom from '@/nodes/overview/OverviewCustom';
import NLLLoss from '@/nodes/overview/loss/Nllloss';

export default class OverviewCanvas extends AbstractCanvas {
  nodeList = [
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
    {
      category: OverviewCategories.Loss,
      nodes: [
        {
          name: OverviewNodes.Nllloss,
          node: NLLLoss,
        },
      ],
    },
  ];

  customNodeType = OverviewCustom;

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(OverviewNodes.ModelNode, Model);
    editor.registerNodeType(OverviewNodes.DataNode, Data);
  }
}
