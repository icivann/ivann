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
    case EditorType.TRAIN: {
      const { trainCanvas } = EditorManager.getInstance();
      canvas = trainCanvas;
      break;
    }
    default:
      break;
  }

  // Guarantee calculate is called on removal of node
  editor.events.removeNode.addListener(editor, () => {
    EditorManager.getInstance().engine.calculate();
  });

  if (canvas) {
    editor.use(canvas.optionPlugin);
    canvas.registerNodes(editor);
  }

  return editor;
}
