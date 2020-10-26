import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import Dense from '@/nodes/model/linear/Dense';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dropout from '@/nodes/model/regularization/Dropout';
import Flatten from '@/nodes/model/reshape/Flatten';
import Custom from '@/nodes/model/custom/Custom';
import Conv1D from '@/nodes/model/conv/Conv1D';
import Conv2D from '@/nodes/model/conv/Conv2D';
import Conv3D from '@/nodes/model/conv/Conv3D';

export default class ModelCanvas extends AbstractCanvas {
  public registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
    editor.registerNodeType(Nodes.Conv1D, Conv1D, Layers.Conv);
    editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
    editor.registerNodeType(Nodes.Conv3D, Conv3D, Layers.Conv);
    editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
    editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
    editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
    editor.registerNodeType(Nodes.Custom, Custom, Layers.Custom);
  }
}
