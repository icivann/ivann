import { EditorModel } from '@/store/editors/types';
import { ModelNodes } from '@/nodes/model/Types';
import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/common/Custom';

export function getEditorIOs(editorModel: EditorModel): { inputs: string[]; outputs: string[] } {
  const inputs: string[] = [];
  const outputs: string[] = [];
  for (const node of editorModel.editor.nodes) {
    switch (node.type) {
      case ModelNodes.InModel:
        inputs.push(node.name);
        break;
      case ModelNodes.OutModel:
        outputs.push(node.name);
        break;
      default:
        break;
    }
  }

  return { inputs, outputs };
}

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

export function deleteNodesFromEditor(editorModel: EditorModel, funcs: ParsedFunction[]) {
  const { editor } = editorModel;
  const { nodes } = editor;

  // Have to loop through copy of nodes, otherwise by removing node, we modify array of nodes st
  // we don't remove duplicate custom nodes
  for (const node of nodes.slice()) {
    if (node instanceof Custom) {
      const func = (node as Custom).getParsedFunction() as ParsedFunction;
      if (funcs.some((f) => f.equals(func))) editor.removeNode(node);
    }
  }
}

export function editNodesFromEditor(editorModel: EditorModel, diff: FuncDiff) {
  const { editor } = editorModel;
  const { nodes } = editor;

  // Have to loop through copy of nodes, otherwise by removing node, we modify array of nodes st
  // we don't remove duplicate custom nodes
  for (const node of nodes.slice()) {
    if (node instanceof Custom) {
      const func = (node as Custom).getParsedFunction() as ParsedFunction;

      // Remove node if deleted
      if (diff.deleted.some((f) => f.equals(func))) editor.removeNode(node);

      // Update node if changed
      const change = diff.changed.find((elem) => elem.oldFunc.name === func.name
        && elem.oldFunc.filename === func.filename);
      if (change) {
        node.setParsedFunction(change.newFunc);
        change.args.added.forEach((arg) => node.addInput(arg));
        change.args.deleted.forEach((arg) => node.remInteface(arg));
      }
    }
  }
}
