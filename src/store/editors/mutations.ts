import { MutationTree } from 'vuex';
import { EditorModel, EditorsState } from '@/store/editors/types';
import { Editor } from '@baklavajs/core';
import newEditor from '@/baklava/Utils';
import EditorType from '@/EditorType';
import EditorManager from '@/EditorManager';
import { loadEditors, Save } from '@/file/EditorAsJson';
import { randomUuid, UUID } from '@/app/util';
import {
  deleteNodeEditor,
  getEditorIOs,
  renameNodeEditor,
  updateNodeConnections,
} from '@/store/editors/utils';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { editNodes, FuncDiff } from '@/store/ManageCodevault';

const editorMutations: MutationTree<EditorsState> = {
  switchEditor(state, { editorType, index }) {
    state.inCodeVault = false;
    state.currEditorType = editorType;
    state.currEditorIndex = index;
    EditorManager.getInstance().resetView();
  },
  newEditor(state, { editorType, name }) {
    state.inCodeVault = false;
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
      default:
        console.log('Attempted to create non existent editor type');
        break;
    }
  },
  renameEditor(state, { editorType, index, name }) {
    let oldName: string | null = null;

    switch (editorType) {
      case EditorType.MODEL:
        oldName = state.modelEditors[index].name;
        state.modelEditors[index].name = name;
        state.editorNames.add(name);
        break;
      case EditorType.DATA:
        oldName = state.dataEditors[index].name;
        state.dataEditors[index].name = name;
        state.editorNames.add(name);
        break;
      default:
        console.log('Attempted to rename non existent editor type');
        break;
    }

    if (oldName !== null) {
      state.editorNames.delete(oldName);

      // Rename all nodes for renamed editor
      renameNodeEditor(state.overviewEditor, oldName, name);
      state.modelEditors.forEach((editor) => {
        if (editor.name !== oldName) renameNodeEditor(editor, oldName!, name);
      });
    }
  },
  deleteEditor(state, { editorType, index }) {
    let name: string | null = null;

    const sameEditorType = state.currEditorType === editorType;
    const diffIndex = state.currEditorIndex - index;
    const deletingCurr = sameEditorType && diffIndex === 0;

    // If deleting current editor, switch to overview
    if (deletingCurr) {
      state.currEditorType = EditorType.OVERVIEW;
      state.currEditorIndex = 0;
    }

    switch (editorType) {
      case EditorType.MODEL:
        name = state.modelEditors[index].name;
        state.modelEditors = state.modelEditors.filter((val, i) => i !== index);
        break;
      case EditorType.DATA:
        name = state.dataEditors[index].name;
        state.dataEditors = state.dataEditors.filter((val, i) => i !== index);
        break;
      default:
        console.log('Attempted to delete non existent editor type');
        break;
    }

    // Maintain same current editor
    if (diffIndex > 0) {
      state.currEditorIndex -= 1;
    }
    EditorManager.getInstance().resetView();

    if (name !== null) {
      state.editorNames.delete(name);

      // Delete all nodes for deleted editor
      deleteNodeEditor(state.overviewEditor, name);
      state.modelEditors.forEach((editor) => {
        if (editor.name !== name) deleteNodeEditor(editor, name!);
      });
    }
  },
  loadEditors(state, file: Save) {
    const editorNames: Set<string> = new Set<string>();

    [state.overviewEditor] = loadEditors(EditorType.OVERVIEW, [file.overviewEditor], editorNames);
    state.modelEditors = loadEditors(EditorType.MODEL, file.modelEditors, editorNames);
    state.dataEditors = loadEditors(EditorType.DATA, file.dataEditors, editorNames);

    state.editorNames = editorNames;

    state.currEditorType = EditorType.OVERVIEW;
    state.currEditorIndex = 0;
  },
  resetState(state) {
    state.overviewEditor = {
      id: randomUuid(),
      name: 'Overview',
      editor: newEditor(EditorType.OVERVIEW),
    };
    state.modelEditors = [];
    state.dataEditors = [];

    state.editorNames = new Set<string>(['Overview']);

    state.currEditorType = EditorType.OVERVIEW;
    state.currEditorIndex = 0;
    state.inCodeVault = false;
  },
  updateNodeInOverview(state, currEditor: EditorModel) {
    // Loop through nodes in currEditor and find differences - TODO: Add type checking here?

    switch (state.currEditorType) {
      case EditorType.MODEL: {
        const { inputs, outputs } = getEditorIOs(currEditor);
        updateNodeConnections(state.overviewEditor, inputs, outputs, currEditor.name);
        state.modelEditors.forEach((editor) => {
          if (editor.name !== currEditor.name) {
            updateNodeConnections(editor, inputs, outputs, currEditor.name);
          }
        });
        break;
      }
      default:
        break;
    }
  },
  enterCodeVault(state) {
    state.inCodeVault = true;
  },
  leaveCodeVault(state) {
    state.inCodeVault = false;
  },
  setErrorsMap(state, errorsMap) {
    state.errorsMap = errorsMap;
  },
  deleteNodes(state, funcs: ParsedFunction[]) {
    editNodes(state.overviewEditor, { deleted: funcs, changed: [] });
    state.modelEditors.forEach((editor) => editNodes(editor, { deleted: funcs, changed: [] }));
    state.dataEditors.forEach((editor) => editNodes(editor, { deleted: funcs, changed: [] }));
  },
  editNodes(state, diff: FuncDiff) {
    editNodes(state.overviewEditor, diff);
    state.modelEditors.forEach((editor) => editNodes(editor, diff));
    state.dataEditors.forEach((editor) => editNodes(editor, diff));
  },
};

export default editorMutations;
