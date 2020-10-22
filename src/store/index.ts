import Vue from 'vue';
import Vuex from 'vuex';
import EditorType from '@/EditorType';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    currEditorType: EditorType.MODEL,
    currEditorIndex: 0,
    overviewEditor: newEditor(EditorType.OVERVIEW), // TODO: Lazy create?
    modelEditors: [newEditor(EditorType.MODEL)],
    dataEditors: [newEditor(EditorType.DATA)], // TODO: Lazy create?
    trainEditors: [newEditor(EditorType.TRAIN)], // TODO: Lazy create?
  },
  getters: {
    currEditorType: (state) => state.currEditorType,
    currEditorIndex: (state) => state.currEditorIndex,
    currEditor: (state, getters) => {
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
          return undefined;
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
    newEditor(state, editorType) {
      const editor: Editor = newEditor(editorType);

      switch (editorType) {
        case EditorType.MODEL:
          state.currEditorType = editorType;
          state.currEditorIndex = state.modelEditors.push(editor) - 1;
          break;
        case EditorType.DATA:
          state.currEditorType = editorType;
          state.currEditorIndex = state.dataEditors.push(editor) - 1;
          break;
        case EditorType.TRAIN:
          state.currEditorType = editorType;
          state.currEditorIndex = state.trainEditors.push(editor) - 1;
          break;
        default:
          break;
      }
    },
  },
  modules: {
  },
});
