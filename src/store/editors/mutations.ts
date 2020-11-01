import { MutationTree } from 'vuex';
import { EditorModel, EditorsState } from '@/store/editors/types';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';
import EditorManager from '@/EditorManager';
import { loadEditor, loadEditors } from '@/file/EditorAsJson';
import { randomUuid, UUID } from '@/app/util';
import Model from '@/nodes/overview/Model';
import editorIOPartition, { NodeIOChange } from '@/nodes/overview/EditorIOUtils';
import { getEditorIOs } from '@/store/editors/utils';

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
        }) - 1;
        break;
      case EditorType.DATA:
        state.currEditorType = editorType;
        state.editorNames.add(name);
        state.currEditorIndex = state.dataEditors.push({
          id,
          name,
          editor,
        }) - 1;
        break;
      case EditorType.TRAIN:
        state.currEditorType = editorType;
        state.editorNames.add(name);
        state.currEditorIndex = state.trainEditors.push({
          id,
          name,
          editor,
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
  resetState(state) {
    state.overviewEditor = {
      id: randomUuid(),
      name: 'Overview',
      editor: newEditor(EditorType.OVERVIEW),
    };
    state.modelEditors = [{
      id: randomUuid(),
      name: 'untitled',
      editor: newEditor(EditorType.MODEL),
    }];
    state.dataEditors = [];
    state.trainEditors = [];

    state.editorNames = new Set<string>(['untitled']);

    state.currEditorType = EditorType.MODEL;
    state.currEditorIndex = 0;
  },
  updateNodeInOverview(state, currEditor: EditorModel) {
    // Loop through nodes in currEditor and find differences
    // inputs, outputs, ...
    // TODO: Add type checking here?
    const { inputs, outputs } = getEditorIOs(currEditor);

    // Loop through nodes in overview editor
    // find corresponding node for currEditor and update
    const { nodes } = state.overviewEditor.editor;
    const overviewNodes = nodes.filter((node) => node.name === currEditor.name) as Model[];
    if (overviewNodes.length > 0) {
      const { inputs: oldInputs, outputs: oldOutputs } = overviewNodes[0].getCurrentIO();
      const inputChange: NodeIOChange = editorIOPartition(inputs, oldInputs);
      const outputChange: NodeIOChange = editorIOPartition(outputs, oldOutputs);
      for (const overviewNode of overviewNodes) {
        overviewNode.updateIO(inputChange, outputChange);
      }
    }
  },
};

export default editorMutations;
