import editorIOPartition from '@/nodes/overview/EditorIOUtils';

describe('partitioning on EditorIO[]', () => {
  it('returns empties for empty array', () => {
    const empty1: string[] = [];
    const empty2: string[] = [];

    const expected = { added: [], removed: [] };
    return expect(editorIOPartition(empty1, empty2)).toStrictEqual(expected);
  });

  it('returns empties for singles', () => {
    const singleUpdated: string[] = ['hello'];
    const singledOld: string[] = ['hello'];

    const expected = { added: [], removed: [] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns correct added and removed', () => {
    const singleUpdated: string[] = ['hello'];
    const singledOld: string[] = ['notAnInterface'];

    const expected = { added: ['hello'], removed: ['notAnInterface'] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns no added but one removed', () => {
    const singleUpdated: string[] = ['hello'];
    const singledOld: string[] = ['hello', 'nope'];

    const expected = { added: [], removed: ['nope'] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });

  it('returns one added but no removed', () => {
    const singleUpdated: string[] = ['hello', 'nope'];
    const singledOld: string[] = ['hello'];

    const expected = { added: ['nope'], removed: [] };
    return expect(editorIOPartition(singleUpdated, singledOld)).toStrictEqual(expected);
  });
});
