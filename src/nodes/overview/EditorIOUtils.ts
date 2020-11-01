export default function editorIOPartition(updated: string[], old: string[]): {
  added: string[];
  removed: string[];
} {
  const added: string[] = [];
  const removed: string[] = [];

  for (const io of updated) {
    if (!old.some((obj) => obj === io)) {
      added.push(io);
    }
  }

  for (const io of old) {
    if (!updated.some((obj) => obj === io)) {
      removed.push(io);
    }
  }

  return { added, removed };
}
