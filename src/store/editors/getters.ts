import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { EditorModels, EditorsState } from '@/store/editors/types';
import EditorType from '@/EditorType';
import { Nodes } from '@/nodes/model/Types';
import { SaveWithNames } from '@/file/EditorAsJson';

const editorGetters: GetterTree<EditorsState, RootState> = {
  currEditorType: (state) => state.currEditorType,
  currEditorIndex: (state) => state.currEditorIndex,
  editorNames: (state) => state.editorNames,
  allEditorModels: (state): EditorModels => ({
    overviewEditor: state.overviewEditor,
    modelEditors: state.modelEditors,
    dataEditors: state.dataEditors,
    trainEditors: state.trainEditors,
  }),
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
  editorIONames: (state, getters) => {
    const names: Set<string> = new Set<string>();
    for (const node of getters.currEditorModel.editor.nodes) {
      if (node.type === Nodes.InModel || node.type === Nodes.OutModel) {
        names.add(node.name);
      }
    }
    return names;
  },
  saveWithNames: (state): SaveWithNames => ({
    overviewEditor: state.overviewEditor.name,
    modelEditors: state.modelEditors.map((editor) => editor.name),
    dataEditors: state.dataEditors.map((editor) => editor.name),
    trainEditors: state.trainEditors.map((editor) => editor.name),
  }),
};

export default editorGetters;
