import { EditorModel } from '@/store/editors/types';
import { ModelNodes } from '@/nodes/model/Types';
import { DataNodes } from '@/nodes/data/Types';
import editorIOPartition, { NodeIOChange } from '@/nodes/overview/EditorIOUtils';

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

export function updateNodeConnections(
  editorModel: EditorModel,
  inputs: string[],
  outputs: string[],
  currEditorName: string,
): void {
  const { nodes } = editorModel.editor;
  // Loop through nodes in editor
  // find corresponding node for currEditor and update
  const modelNodes = nodes.filter((node) => node.name === currEditorName) as any[];
  if (modelNodes.length > 0) {
    const { inputs: oldInputs, outputs: oldOutputs } = modelNodes[0].getCurrentIO();
    const inputChange: NodeIOChange = editorIOPartition(inputs, oldInputs);
    const outputChange: NodeIOChange = editorIOPartition(outputs, oldOutputs);
    for (const modelNode of modelNodes) {
      modelNode.updateIO(inputChange, outputChange);
    }
  }
}
