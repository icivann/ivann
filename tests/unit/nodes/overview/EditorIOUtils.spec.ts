import editorIOPartition from '@/nodes/overview/EditorIOUtils';
import { EditorIO } from '@/store/editors/types';

describe('equality on EditorIO[]', () => {
  it('returns empties for empty array', () => {
    const empty1: EditorIO[] = [];
    const empty2: EditorIO[] = [];

    const expected = { added: [], removed: [] };
    return expect(editorIOPartition(empty1, empty2)).toStrictEqual(expected);
  });

  it('returns empties for singles', () => {
    const singleUpdated: EditorIO[] = [{ name: 'hello' }];
    const singledOld: EditorIO[] = [{ name: 'hello' }];

    const expected = { added: [], removed: [] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns correct added and removed', () => {
    const singleUpdated: EditorIO[] = [{ name: 'hello' }];
    const singledOld: EditorIO[] = [{ name: 'notAnInterface' }];

    const expected = { added: [{ name: 'hello' }], removed: [{ name: 'notAnInterface' }] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns no added but one removed', () => {
    const singleUpdated: EditorIO[] = [{ name: 'hello' }];
    const singledOld: EditorIO[] = [{ name: 'hello' }, { name: 'nope' }];

    const expected = { added: [], removed: [{ name: 'nope' }] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns one added but no removed', () => {
    const singleUpdated: EditorIO[] = [{ name: 'hello' }, { name: 'nope' }];
    const singledOld: EditorIO[] = [{ name: 'hello' }];

    const expected = { added: [{ name: 'nope' }], removed: [] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });
});
