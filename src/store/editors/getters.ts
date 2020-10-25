import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { EditorsState } from '@/store/editors/types';
import EditorType from '@/EditorType';

const editorGetters: GetterTree<EditorsState, RootState> = {
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
};

export default editorGetters;
