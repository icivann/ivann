import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import { UUID } from '@/app/util';
import IrError from '@/app/ir/checking/irError';

export interface EditorModel {
  id: UUID;
  name: string;
  editor: Editor;
}

export interface EditorModels {
  overviewEditor: EditorModel;
  modelEditors: EditorModel[];
  dataEditors: EditorModel[];
}

export interface EditorsState extends EditorModels {
  currEditorType: EditorType;
  currEditorIndex: number;
  editorNames: Set<string>;
  inCodeVault: boolean;
  errorsMap: Map<string, IrError[]>;
}
