import { EditorModel } from '@/store/editors/types';
import Custom from '@/nodes/common/Custom';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { saveEditor } from '@/file/EditorAsJson';

export interface FuncDiff {
  deleted: ParsedFunction[];
  changed: ({
    oldFunc: ParsedFunction;
    newFunc: ParsedFunction;
    args: { deleted: string[]; added: string[] };
  })[];
}

/*
Compares two ParsedFunction arrays
Returns object containing:
  deleted:
    - If function is in oldFunc but not in newFunc
  changed:
    - If function in oldFunc is present in newFunc but has been changed
    - Changed if same name but different args and/or body
    - Contains:
        oldFunc - function that was changed
        newFunc - function it has become
        args:
          deleted - args that was in oldFunc but not in newFunc
          added - args that was not in oldFunc but is in newFunc
 */
export function funcsDiff(
  oldFuncs: ParsedFunction[], newFuncs: ParsedFunction[],
): FuncDiff {
  const deleted: ParsedFunction[] = [];
  const changed: ({
    oldFunc: ParsedFunction;
    newFunc: ParsedFunction;
    args: { deleted: string[]; added: string[] };
  })[] = [];

  oldFuncs.forEach((oldFunc) => {
    const isPresent = newFuncs.some((newFunc) => newFunc.name === oldFunc.name);

    // If not present, then has been deleted
    // Else is present and must check if has changed
    if (!isPresent) {
      deleted.push(oldFunc);
      console.log(`deleted ${oldFunc.name}`);
    } else {
      const newFunc = newFuncs.find((newFunc) => newFunc.name === oldFunc.name);

      // Do not add to changed array if same function
      if (newFunc === undefined || oldFunc.equals(newFunc)) return;

      const deletedArgs: string[] = oldFunc.args.filter((arg) => !newFunc.args.includes(arg));
      const addedArgs: string[] = newFunc.args.filter((arg) => !oldFunc.args.includes(arg));
      changed.push({
        oldFunc,
        newFunc,
        args: { deleted: deletedArgs, added: addedArgs },
      });
    }
  });

  return { deleted, changed };
}

export function editNodes(editorModel: EditorModel, diff: FuncDiff) {
  const { name, editor } = editorModel;
  const { nodes } = editor;

  // Track if any nodes have changed, if so, need to update local storage
  let needToSave = false;

  // Have to loop through copy of nodes, otherwise by removing node, we modify array of nodes st
  // we don't remove duplicate custom nodes
  for (const node of nodes.slice()) {
    if (node instanceof Custom) {
      const func = (node as Custom).getParsedFunction() as ParsedFunction;

      // Remove node if deleted
      if (diff.deleted.some((f) => f.equals(func))) {
        needToSave = true;
        editor.removeNode(node);
      }

      // Update node if changed
      const change = diff.changed.find((elem) => elem.oldFunc.name === func.name
        && elem.oldFunc.filename === func.filename);
      if (change) {
        needToSave = true;
        node.setParsedFunction(change.newFunc);
        change.args.added.forEach((arg) => node.addInput(arg));
        change.args.deleted.forEach((arg) => node.remInteface(arg));
      }
    }
  }

  if (needToSave) {
    const saved = saveEditor(editorModel);
    localStorage.setItem(`unsaved-editor-${name}`, JSON.stringify(saved));
  }
}
