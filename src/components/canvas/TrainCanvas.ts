import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';

export default class TrainCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    console.log('No nodes registered');
  }
}
