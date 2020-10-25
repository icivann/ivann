import { MutationTree } from 'vuex';
import { EditorsState } from '@/store/editors/types';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';

const editorMutations: MutationTree<EditorsState> = {
  switchEditor(state, { editorType, index }) {
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
};

export default editorMutations;
