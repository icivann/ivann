import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';

import { Nodes, NodeTypes } from '@/nodes/data/Types';

import InData from '@/nodes/data/InData';
import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';

export default class DataCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.InData, InData, NodeTypes.IO);
    editor.registerNodeType(Nodes.OutData, OutData, NodeTypes.IO);
    editor.registerNodeType(Nodes.ToTensor, ToTensor, NodeTypes.Transform);
    editor.registerNodeType(Nodes.Grayscale, Grayscale, NodeTypes.Transform);
  }
}
