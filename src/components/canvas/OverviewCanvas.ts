import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import Model from '@/nodes/overview/Model';
import TrainClassifier from '@/nodes/train/TrainClassifier';
import { OverviewNodes, OverviewLayers } from '@/nodes/Types';
import Adadelta from '@/nodes/train/optimizers/Adadelta';

export default class OverviewCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(OverviewNodes.Model, Model);
    editor.registerNodeType(OverviewNodes.Model, Model);
    editor.registerNodeType(OverviewNodes.TrainClassifier, TrainClassifier, OverviewLayers.Train);
    editor.registerNodeType(OverviewNodes.Adadelta, Adadelta, OverviewLayers.Optimizers);
  }
}
