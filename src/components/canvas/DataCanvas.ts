import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';

export default class DataCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    // editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
  }
}
