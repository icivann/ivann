import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import { UUID } from '@/app/util';
import Custom from '@/nodes/model/custom/Custom';

export interface EditorModel {
  id: UUID;
  name: string;
  editor: Editor;
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
  inCodeVault: boolean;
  nodeTriggeringCodeVault?: Custom;
}
