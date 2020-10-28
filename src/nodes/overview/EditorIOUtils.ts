export default function editorIOEquals(o1: any, o2: any): boolean {
  if (!Array.isArray(o1) || !Array.isArray(o2) || o1.length !== o2.length) {
    return false;
  }

  for (let i = 0; i < o1.length; i += 1) {
    const { name: name1 } = o1[i];
    const { name: name2 } = o2[i];
    if (name1 === undefined || name1 !== name2) {
      return false;
    }
  }
  return true;
}
