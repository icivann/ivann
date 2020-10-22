import Vue from 'vue';
import Vuex, { StoreOptions } from 'vuex';
import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';
import { RootState } from '@/store/Types';

Vue.use(Vuex);

const store: StoreOptions<RootState> = {
  state: {
    currEditorType: EditorType.MODEL,
    currEditorIndex: 0,
    overviewEditor: {
      name: 'Overview',
      editor: newEditor(EditorType.OVERVIEW), // TODO: Lazy create?
    },
    modelEditors: [
      {
        name: 'untitled',
        editor: newEditor(EditorType.MODEL),
      },
    ],
    dataEditors: [],
    trainEditors: [],
  },
  getters: {
    currEditorType: (state) => state.currEditorType,
    currEditorIndex: (state) => state.currEditorIndex,
    currEditorModel: (state, getters) => {
      const index = getters.currEditorIndex;
      switch (getters.currEditorType) {
        case EditorType.OVERVIEW:
          return state.overviewEditor;
        case EditorType.MODEL:
          return state.modelEditors[index];
        case EditorType.DATA:
          return state.dataEditors[index];
        case EditorType.TRAIN:
          return state.trainEditors[index];
        default:
          return {};
      }
    },
    modelEditors: (state) => state.modelEditors,
    dataEditors: (state) => state.dataEditors,
    trainEditors: (state) => state.trainEditors,
    overviewEditor: (state) => state.overviewEditor,
    modelEditor: (state) => (index: number) => state.modelEditors[index],
    dataEditor: (state) => (index: number) => state.dataEditors[index],
    trainEditor: (state) => (index: number) => state.trainEditors[index],
  },
  actions: {
  },
  mutations: {
    switchEditor(state, { editorType, index }) { // TODO: index
      state.currEditorType = editorType;
      state.currEditorIndex = index;
    },
    newEditor(state, { editorType, name }) {
      const editor: Editor = newEditor(editorType);

      switch (editorType) {
        case EditorType.MODEL:
          state.currEditorType = editorType;
          state.currEditorIndex = state.modelEditors.push({ name, editor }) - 1;
          break;
        case EditorType.DATA:
          state.currEditorType = editorType;
          state.currEditorIndex = state.dataEditors.push({ name, editor }) - 1;
          break;
        case EditorType.TRAIN:
          state.currEditorType = editorType;
          state.currEditorIndex = state.trainEditors.push({ name, editor }) - 1;
          break;
        default:
          break;
      }
    },
  },
  modules: {
  },
};

export default new Vuex.Store<RootState>(store);
