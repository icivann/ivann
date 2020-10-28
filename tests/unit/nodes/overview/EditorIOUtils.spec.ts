import editorIOEquals from '@/nodes/overview/EditorIOUtils';
import { EditorIO } from '@/store/editors/types';

describe('equality on EditorIO[]', () => {
  it('passes for empty array', () => {
    const empty1: EditorIO[] = [];
    const empty2: EditorIO[] = [];

    return expect(editorIOEquals(empty1, empty2)).toBe(true);
  });

  it('fails for other objects', () => {
    const o1 = 'not an array';
    const o2: EditorIO[] = [];

    return expect(editorIOEquals(o1, o2)).toBe(false);
  });

  it('passes for singles', () => {
    const single1: EditorIO[] = [{ name: 'hello' }];
    const single2: EditorIO[] = [{ name: 'hello' }];

    return expect(editorIOEquals(single1, single2)).toBe(true);
  });

  it('fails for singles different names', () => {
    const single1: EditorIO[] = [{ name: 'hello' }];
    const single2: EditorIO[] = [{ name: 'notAnInterface' }];

    return expect(editorIOEquals(single1, single2)).toBe(false);
  });

  it('fails for different lengths', () => {
    const single1: EditorIO[] = [{ name: 'hello' }];
    const single2: EditorIO[] = [{ name: 'hello' }, { name: 'nope' }];

    return expect(editorIOEquals(single1, single2)).toBe(false);
  });

  it('passes for doubles', () => {
    const double1: EditorIO[] = [{ name: 'hello' }, { name: 'nope' }];
    const double2: EditorIO[] = [{ name: 'hello' }, { name: 'nope' }];

    return expect(editorIOEquals(double1, double2)).toBe(true);
  });
});
