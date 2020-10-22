import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Layers, Nodes } from '@/nodes/model/Types';
import Dense from '@/nodes/model/linear/Dense';
import Conv2D from '@/nodes/model/conv/Conv2D';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dropout from '@/nodes/model/regularization/Dropout';
import Flatten from '@/nodes/model/reshape/Flatten';
import Custom from '@/nodes/model/custom/Custom';
import { Editor } from '@baklavajs/core';

export default class ModelCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
    editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
    editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
    editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
    editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
    editor.registerNodeType(Nodes.Custom, Custom, Layers.Custom);
  }
}
