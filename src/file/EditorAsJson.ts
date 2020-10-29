import { EditorModel } from '@/store/editors/types';
import EditorType from '@/EditorType';
import newEditor from '@/baklava/Utils';
import { Editor } from '@baklavajs/core';
import EditorManager from '@/EditorManager';
import { randomUuid, UUID } from '@/app/util';

export const FILENAME = 'ivann';

export function saveEditor(editor: EditorModel) {
  return {
    name: editor.name,
    editorState: editor.editor.save(),
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  editorType: EditorType, editorSaved: any, names: Set<string>,
): EditorModel {
  const { name, editorState } = editorSaved;
  const id: UUID = randomUuid();

  names.add(name);

  const editor: Editor = newEditor(editorType);
  editor.use(EditorManager.getInstance().viewPlugin);
  editor.load(editorState);

  return { id, name, editor };
}

export function loadEditors(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  editorType: EditorType, editorsSaved: any, names: Set<string>,
): EditorModel[] {
  const editorsLoaded = [];
  for (const editorSaved of editorsSaved) {
    editorsLoaded.push(loadEditor(editorType, editorSaved, names));
  }

  return editorsLoaded;
}
