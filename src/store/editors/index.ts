import { Module } from 'vuex';
import { RootState } from '@/store/types';
import EditorType from '@/EditorType';
import newEditor from '@/baklava/Utils';
import editorGetters from './getters';
import editorMutations from './mutations';
import { EditorsState } from './types';

export const editorState: EditorsState = {
  currEditorType: EditorType.MODEL,
  currEditorIndex: 0,
  overviewEditor: {
    name: 'Overview',
    editor: newEditor(EditorType.OVERVIEW), // TODO: Lazy create?
  },
  modelEditors: [
    {
      name: 'untitled 0',
      editor: newEditor(EditorType.MODEL),
    },
  ],
  dataEditors: [],
  trainEditors: [],
};

export const editors: Module<EditorsState, RootState> = {
  namespaced: false,
  state: editorState,
  getters: editorGetters,
  actions: undefined,
  mutations: editorMutations,
};
