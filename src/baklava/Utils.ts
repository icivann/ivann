import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import EditorManager from '@/EditorManager';
import AbstractCanvas from '@/components/canvas/AbstractCanvas';

export default function newEditor(editorType: EditorType) {
  let canvas: AbstractCanvas | undefined;
  const editor = new Editor();

  switch (editorType) {
    case EditorType.OVERVIEW: {
      const { overviewCanvas } = EditorManager.getInstance();
      canvas = overviewCanvas;
      break;
    }
    case EditorType.MODEL: {
      const { modelCanvas } = EditorManager.getInstance();
      canvas = modelCanvas;
      break;
    }
    case EditorType.DATA: {
      const { dataCanvas } = EditorManager.getInstance();
      canvas = dataCanvas;
      break;
    }
    default:
      break;
  }

  if (canvas) {
    editor.use(canvas.optionPlugin);
    canvas.registerNodes(editor);
  }

  return editor;
}
