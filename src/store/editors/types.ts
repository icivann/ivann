import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import { UUID } from '@/app/util';

export interface EditorIO {
  name: string;
}

export interface EditorModel {
  id: UUID;
  name: string;
  editor: Editor;
  saved: boolean;
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
