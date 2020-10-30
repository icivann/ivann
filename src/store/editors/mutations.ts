import { MutationTree } from 'vuex';
import { EditorIO, EditorsState } from '@/store/editors/types';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';
import EditorManager from '@/EditorManager';
import { loadEditor, loadEditors } from '@/file/EditorAsJson';
import { randomUuid, UUID } from '@/app/util';
import { Nodes } from '@/nodes/model/Types';

const editorMutations: MutationTree<EditorsState> = {
  switchEditor(state, { editorType, index }) {
    state.currEditorType = editorType;
    state.currEditorIndex = index;
    EditorManager.getInstance().resetView();
  },
  newEditor(state, { editorType, name }) {
    const id: UUID = randomUuid();
    const editor: Editor = newEditor(editorType);

    switch (editorType) {
      case EditorType.MODEL:
        state.currEditorType = editorType;
        state.editorNames.add(name);
        state.currEditorIndex = state.modelEditors.push({
          id,
          name,
          editor,
          inputs: [],
          outputs: [],
          saved: true,
        }) - 1;
        break;
      case EditorType.DATA:
        state.currEditorType = editorType;
        state.editorNames.add(name);
        state.currEditorIndex = state.dataEditors.push({
          id,
          name,
          editor,
          saved: true,
        }) - 1;
        break;
      case EditorType.TRAIN:
        state.currEditorType = editorType;
        state.editorNames.add(name);
        state.currEditorIndex = state.trainEditors.push({
          id,
          name,
          editor,
          saved: true,
        }) - 1;
        break;
      default:
        break;
    }
  },
  loadEditors(state, file) {
    const editorNames: Set<string> = new Set<string>();

    state.overviewEditor = loadEditor(EditorType.OVERVIEW, file.overviewEditor, editorNames);
    state.modelEditors = loadEditors(EditorType.MODEL, file.modelEditors, editorNames);
    state.dataEditors = loadEditors(EditorType.DATA, file.dataEditors, editorNames);
    state.trainEditors = loadEditors(EditorType.TRAIN, file.trainEditors, editorNames);

    state.editorNames = editorNames;

    state.currEditorType = EditorType.MODEL;
    state.currEditorIndex = 0;
  },
  saveModel(state, index) {
    const { editor } = state.modelEditors[index];
    const inputs: EditorIO[] = [];
    const outputs: EditorIO[] = [];
    for (const node of editor.nodes) {
      if (node.type === Nodes.InModel) inputs.push({ name: node.name });
      else if (node.type === Nodes.OutModel) outputs.push({ name: node.name });
    }
    state.modelEditors[index].inputs = inputs;
    state.modelEditors[index].outputs = outputs;
    state.modelEditors[index].saved = true;
  },
  setUnsaved(state) {
    state.modelEditors[state.currEditorIndex].saved = false;
  },
};

export default editorMutations;
