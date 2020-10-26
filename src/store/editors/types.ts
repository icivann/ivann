import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';

export interface EditorModel {
  name: string;
  editor: Editor;
}

export interface EditorsState {
  currEditorType: EditorType;
  currEditorIndex: number;
  editorNames: Set<string>;
  overviewEditor: EditorModel;
  modelEditors: EditorModel[];
  dataEditors: EditorModel[];
  trainEditors: EditorModel[];
}
