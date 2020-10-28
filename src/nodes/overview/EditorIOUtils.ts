import { EditorIO } from '@/store/editors/types';

export default function editorIOPartition(updated: EditorIO[], old: EditorIO[]): {
  added: EditorIO[];
  removed: EditorIO[];
} {
  const added: EditorIO[] = [];
  const removed: EditorIO[] = [];

  for (const io of updated) {
    if (!old.some((obj) => obj.name === io.name)) {
      added.push(io);
    }
  }

  for (const io of old) {
    if (!updated.some((obj) => obj.name === io.name)) {
      removed.push(io);
    }
  }

  return { added, removed };
}
