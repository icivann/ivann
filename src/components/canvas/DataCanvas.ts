import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';

export default class DataCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    console.log('No nodes registered');
  }
}
