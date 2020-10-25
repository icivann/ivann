import { EditorModel, EditorsState } from '@/store/editors/types';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';

export const mockOverviewEditor: EditorModel = {
  name: 'Overview',
  editor: newEditor(EditorType.OVERVIEW),
};

export const mockModelEditors: EditorModel[] = [
  {
    name: 'name0',
    editor: newEditor(EditorType.MODEL),
  },
  {
    name: 'name1',
    editor: newEditor(EditorType.MODEL),
  },
  {
    name: 'name2',
    editor: newEditor(EditorType.MODEL),
  },
];

export const mockEditorState: EditorsState = {
  currEditorType: EditorType.MODEL,
  currEditorIndex: 2,
  overviewEditor: mockOverviewEditor,
  modelEditors: mockModelEditors,
  dataEditors: [],
  trainEditors: [],
};
