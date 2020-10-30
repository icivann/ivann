import { EditorModel, EditorsState } from '@/store/editors/types';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';
import { randomUuid } from '@/app/util';

export const mockOverviewEditor: EditorModel = {
  id: randomUuid(),
  name: 'Overview',
  editor: newEditor(EditorType.OVERVIEW),
  saved: true,
};

export const mockModelEditors: EditorModel[] = [
  {
    id: randomUuid(),
    name: 'name0',
    editor: newEditor(EditorType.MODEL),
    saved: true,
  },
  {
    id: randomUuid(),
    name: 'name1',
    editor: newEditor(EditorType.MODEL),
    saved: true,
  },
  {
    id: randomUuid(),
    name: 'name2',
    editor: newEditor(EditorType.MODEL),
    saved: true,
  },
];

export const mockEditorState: EditorsState = {
  currEditorType: EditorType.MODEL,
  currEditorIndex: 2,
  editorNames: new Set<string>(['name0', 'name1', 'name2']),
  overviewEditor: mockOverviewEditor,
  modelEditors: mockModelEditors,
  dataEditors: [],
  trainEditors: [],
};
