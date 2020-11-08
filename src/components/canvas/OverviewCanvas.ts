import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import Model from '@/nodes/overview/Model';
import TrainClassifier from '@/nodes/train/TrainClassifier';
import { OverviewNodes, OverviewLayers } from '@/nodes/model/Types';

export default class OverviewCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(OverviewNodes.Model, Model);
    editor.registerNodeType(OverviewNodes.Model, Model);
    editor.registerNodeType(OverviewNodes.TrainClassifier, TrainClassifier, OverviewLayers.Train);
  }
}
