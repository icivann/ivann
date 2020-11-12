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
      default:
        return {};
    }
  },
  modelEditors: (state) => state.modelEditors,
  dataEditors: (state) => state.dataEditors,
  overviewEditor: (state) => state.overviewEditor,
  modelEditor: (state) => (index: number) => state.modelEditors[index],
  dataEditor: (state) => (index: number) => state.dataEditors[index],
  editor: (state) => (editorType: EditorType, index: number) => {
    switch (editorType) {
      case EditorType.OVERVIEW:
        return state.overviewEditor;
      case EditorType.MODEL:
        return state.modelEditors[index];
      case EditorType.DATA:
        return state.dataEditors[index];
      default:
        return {};
    }
  },
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
  }),
  inCodeVault: (state) => state.inCodeVault,
};

export default editorGetters;
