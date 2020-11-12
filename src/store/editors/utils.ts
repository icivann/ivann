import { EditorModel } from '@/store/editors/types';
import { ModelNodes } from '@/nodes/model/Types';

export function getEditorIOs(editorModel: EditorModel): { inputs: string[]; outputs: string[] } {
  const inputs: string[] = [];
  const outputs: string[] = [];
  for (const node of editorModel.editor.nodes) {
    switch (node.type) {
      case ModelNodes.InModel:
        inputs.push(node.name);
        break;
      case ModelNodes.OutModel:
        outputs.push(node.name);
        break;
      default:
        break;
    }
  }

  return { inputs, outputs };
}
