import editorMutations from '@/store/editors/mutations';
import EditorType from '@/EditorType';
import { mockEditorState } from './mockData';

describe('editorMutations', () => {
  describe('switchEditor', () => {
    test('sets editorType and index', () => {
      const editorType: EditorType = EditorType.DATA;
      const index = 7;
      const payload = { editorType, index };
      editorMutations.switchEditor(mockEditorState, payload);

      expect(mockEditorState).toMatchObject({
        currEditorType: editorType,
        currEditorIndex: index,
      });
    });
  });

  describe('newEditor', () => {
    test('does not change current editor if unknown editorType', () => {
      const editorType: EditorType = 1000;
      const index = 4;
      const payload = { editorType, index };
      editorMutations.newEditor(mockEditorState, payload);

      expect(mockEditorState.currEditorType).not.toEqual(editorType);
      expect(mockEditorState.currEditorIndex).not.toEqual(index);
    });

    test('creates new editor and sets it to be current', () => {
      const editorType: EditorType = EditorType.DATA;
      const name = 'mockName';
      const payload = { editorType, name };

      const currNumDataEditors = mockEditorState.dataEditors.length;
      editorMutations.newEditor(mockEditorState, payload);
      const newNumDataEditors = mockEditorState.dataEditors.length;

      expect(newNumDataEditors - currNumDataEditors).toEqual(1);

      expect(mockEditorState.editorNames.has(name)).toBe(true);
      expect(mockEditorState).toMatchObject({
        currEditorType: editorType,
        currEditorIndex: newNumDataEditors - 1,
      });
    });
  });
});
