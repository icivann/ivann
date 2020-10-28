import Input from '@/nodes/model/io/Input';
import { EditorModel } from '@/store/editors/types';

export function inputAdded(node: Input, currEditorModel?: EditorModel): void {
  if (!currEditorModel || !currEditorModel.inputs) return;
  const count = currEditorModel.inputs.length;
  const inputName = `Input${count}`;
  currEditorModel.inputs.push({ name: inputName });
  node.setName(inputName);
}

export function outputAdded(node: Input, currEditorModel?: EditorModel): void {
  if (!currEditorModel || !currEditorModel.outputs) return;
  const count = currEditorModel.outputs.length;
  const outputName = `Output${count}`;
  currEditorModel.outputs.push({ name: outputName });
  node.setName(outputName);
}
