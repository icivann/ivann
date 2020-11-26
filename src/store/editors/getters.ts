import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { EditorModels, EditorsState } from '@/store/editors/types';
import EditorType from '@/EditorType';
import { ModelNodes } from '@/nodes/model/Types';
import { SaveWithNames } from '@/file/EditorAsJson';
import { DataNodes } from '@/nodes/data/Types';
import { FuncDiff, usedNodes } from '@/store/ManageCodevault';

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
      if (node.type === ModelNodes.InModel
        || node.type === ModelNodes.OutModel
        || node.type === DataNodes.InData) {
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
  errorsMap: (state) => state.errorsMap,
  usedNodes: (state) => (diff: FuncDiff) => {
    const used = [];
    used.push(usedNodes(state.overviewEditor, diff));
    state.modelEditors.forEach((editor) => used.push(usedNodes(editor, diff)));
    state.dataEditors.forEach((editor) => used.push(usedNodes(editor, diff)));
    return used.filter((use) => use.deleted.length > 0 || use.changed.length > 0);
  },
};

export default editorGetters;
