import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';

export interface EditorIO {
  name: string;
}

export interface EditorModel {
  name: string;
  editor: Editor;
  inputs?: EditorIO[];
  outputs?: EditorIO[];
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
