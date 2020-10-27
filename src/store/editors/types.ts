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

export interface EditorModels {
  overviewEditor: EditorModel;
  modelEditors: EditorModel[];
  dataEditors: EditorModel[];
  trainEditors: EditorModel[];
}

export interface EditorsState extends EditorModels {
  currEditorType: EditorType;
  currEditorIndex: number;
  editorNames: Set<string>;
}
