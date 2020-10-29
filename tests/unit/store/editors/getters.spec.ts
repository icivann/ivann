import editorGetters from '@/store/editors/getters';
import { EditorsState } from '@/store/editors/types';
import { RootState } from '@/store/types';
import { mockEditorState, mockModelEditors } from './mockData';

function getCurrEditorType(state: Partial<EditorsState>) {
  return editorGetters.currEditorType(
    state as EditorsState,
    undefined,
    {} as RootState,
    undefined,
  );
}

function getCurrEditorIndex(state: Partial<EditorsState>) {
  return editorGetters.currEditorIndex(
    state as EditorsState,
    undefined,
    {} as RootState,
    undefined,
  );
}

function getCurrEditorModel(state: Partial<EditorsState>) {
  const currEditorType = getCurrEditorType(state);
  const currEditorIndex = getCurrEditorIndex(state);
  return editorGetters.currEditorModel(
    state as EditorsState,
    { currEditorType, currEditorIndex },
    {} as RootState,
    undefined,
  );
}

describe('editorGetters', () => {
  describe('currEditorModel', () => {
    test('returns empty object when unknown currEditorType', () => {
      const state: Partial<EditorsState> = {
        currEditorIndex: 1,
        currEditorType: 1000,
      };

      const currEditorModel = getCurrEditorModel(state);

      expect(currEditorModel).toEqual({});
    });

    test('returns currEditorModel using current index and type', () => {
      const currEditorModel = getCurrEditorModel(mockEditorState);

      expect(currEditorModel).toBe(mockModelEditors[2]);
    });
  });

  describe('modelEditor', () => {
    test('returns undefined when index is out of bounds', () => {
      const state: Partial<EditorsState> = {
        modelEditors: [],
      };
      const modelEditorGetter = editorGetters.modelEditor(
        state as EditorsState,
        undefined,
        {} as RootState,
        undefined,
      );

      expect(modelEditorGetter(-1)).toBeUndefined();
      expect(modelEditorGetter(1)).toBeUndefined();
    });

    test('returns modelEditor of given index', () => {
      const index = 1;

      const editor = editorGetters.modelEditor(
        mockEditorState,
        undefined,
        {} as RootState,
        undefined,
      )(index);
      expect(editor).toBe(mockEditorState.modelEditors[index]);
    });
  });
});
