import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import DatasetInput from '@/nodes/train/DatasetInput';
import Loss from '@/nodes/train/Loss';
import ToDevice from '@/nodes/train/ToDevice';
import { Nodes } from '@/nodes/train/Types';
import { Editor } from '@baklavajs/core';

export default class TrainCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.DatasetInput, DatasetInput);
    editor.registerNodeType(Nodes.ToDevice, ToDevice);
    editor.registerNodeType(Nodes.Loss, Loss);
    // TODO: finish other nodes
    // editor.registerNodeType(Nodes.Backpropagate, Backpropagate);
    // editor.registerNodeType(Nodes.ZeroGrad, ZeroGrad);
    // editor.registerNodeType(Nodes.Model, Model);
  }
}
