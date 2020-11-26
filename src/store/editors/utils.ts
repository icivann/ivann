import { EditorModel } from '@/store/editors/types';
import { ModelNodes } from '@/nodes/model/Types';
import { DataNodes } from '@/nodes/data/Types';

export function getEditorIOs(editorModel: EditorModel): { inputs: string[]; outputs: string[] } {
  const inputs: string[] = [];
  const outputs: string[] = [];
  for (const node of editorModel.editor.nodes) {
    switch (node.type) {
      case DataNodes.LoadCsv:
      case DataNodes.LoadImages:
      case ModelNodes.InModel:
        inputs.push(node.name);
        break;
      case DataNodes.OutData:
      case ModelNodes.OutModel:
        outputs.push(node.name);
        break;
      default:
        break;
    }
  }

  return { inputs, outputs };
}
