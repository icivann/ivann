import { Module } from 'vuex';
import { randomUuid } from '@/app/util';
import { RootState } from '@/store/types';
import EditorType from '@/EditorType';
import newEditor from '@/baklava/Utils';
import editorGetters from './getters';
import editorMutations from './mutations';
import { EditorsState } from './types';

export const editorState: EditorsState = {
  currEditorType: EditorType.OVERVIEW,
  currEditorIndex: 0,
  editorNames: new Set<string>(['Overview']),
  overviewEditor: {
    // TODO: Map of editors used in overview to their nodes?
    id: randomUuid(),
    name: 'Overview',
    editor: newEditor(EditorType.OVERVIEW),
  },
  modelEditors: [],
  dataEditors: [],
  inCodeVault: false,
  errorsMap: new Map(),
  ioNames: new Set<string>(),
};

export const editors: Module<EditorsState, RootState> = {
  namespaced: false,
  state: editorState,
  getters: editorGetters,
  actions: undefined,
  mutations: editorMutations,
};
