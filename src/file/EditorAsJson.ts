/* eslint-disable max-classes-per-file */
import { EditorModel } from '@/store/editors/types';
import EditorType from '@/EditorType';
import newEditor from '@/baklava/Utils';
import { Editor } from '@baklavajs/core';
import EditorManager from '@/EditorManager';
import { randomUuid, UUID } from '@/app/util';
import { IState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';

export const FILENAME = 'ivann';

export function saveEditor(editor: EditorModel): EditorSave {
  return {
    name: editor.name,
    state: editor.editor.save(),
  };
}

export function saveEditors(editors: EditorModel[]) {
  const editorsSaved = [];
  for (const editor of editors) {
    editorsSaved.push(saveEditor(editor));
  }

  return editorsSaved;
}

export function loadEditor(
  editorType: EditorType, editorSaved: EditorSave, names: Set<string>,
): EditorModel {
  const { name, state } = editorSaved;
  const id: UUID = randomUuid();

  names.add(name);

  const editor: Editor = newEditor(editorType);
  editor.use(EditorManager.getInstance().viewPlugin);
  editor.use(EditorManager.getInstance().engine);
  editor.load(state);

  return { id, name, editor };
}

export function loadEditors(
  editorType: EditorType, editorsSaved: EditorSave[], names: Set<string>,
): EditorModel[] {
  const editorsLoaded = [];
  for (const editorSaved of editorsSaved) {
    editorsLoaded.push(loadEditor(editorType, editorSaved, names));
  }

  return editorsLoaded;
}

export interface EditorSave {
  name: string;
  state: IState;
}

export interface Save {
  overviewEditor: EditorSave;
  modelEditors: EditorSave[];
  dataEditors: EditorSave[];
  trainEditors: EditorSave[];
}

export class SaveWithNames {
  constructor(
    public readonly overviewEditor: string,
    public readonly modelEditors: string[],
    public readonly dataEditors: string[],
    public readonly trainEditors: string[],
  ) {}
}
