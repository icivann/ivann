import { EditorModel } from '@/store/editors/types';
import { Nodes } from '@/nodes/model/Types';

export function getEditorIOs(editorModel: EditorModel): { inputs: string[]; outputs: string[] } {
  const inputs: string[] = [];
  const outputs: string[] = [];
  for (const node of editorModel.editor.nodes) {
    switch (node.type) {
      case Nodes.InModel:
        inputs.push(node.name);
        break;
      case Nodes.OutModel:
        outputs.push(node.name);
        break;
      default:
        break;
    }
  }

  return { inputs, outputs };
}
