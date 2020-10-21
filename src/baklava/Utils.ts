import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import Dense from '@/nodes/model/linear/Dense';
import Conv2D from '@/nodes/model/conv/Conv2D';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dropout from '@/nodes/model/regularization/Dropout';
import Flatten from '@/nodes/model/reshape/Flatten';
import EditorManager from '@/EditorManager';

export default function newEditor(editorType: EditorType) {
  const editor = new Editor();

  switch (editorType) {
    case EditorType.OVERVIEW: {
      const { overviewCanvas } = EditorManager.getInstance();
      editor.use(overviewCanvas.optionPlugin);
      // editor.use(overviewCanvas.viewPlugin);
      break;
    }
    case EditorType.MODEL: {
      const { modelCanvas } = EditorManager.getInstance();
      editor.use(modelCanvas.optionPlugin);
      // editor.use(modelCanvas.viewPlugin);

      editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
      editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
      editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
      editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
      editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
      break;
    }
    case EditorType.DATA: {
      const { dataCanvas } = EditorManager.getInstance();
      editor.use(dataCanvas.optionPlugin);
      // editor.use(dataCanvas.viewPlugin);
      break;
    }
    case EditorType.TRAIN: {
      const { trainCanvas } = EditorManager.getInstance();
      editor.use(trainCanvas.optionPlugin);
      // editor.use(trainCanvas.viewPlugin);
      break;
    }
    default:
      break;
  }

  return editor;
}
