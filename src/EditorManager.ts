import { Editor } from '@baklavajs/core';
import EditorType from '@/EditorType';

export default class EditorManager {
  get overviewBaklavaEditor(): Editor {
    return this.overview;
  }
  get dataBaklavaEditor(): Editor {
    return this.data;
  }
  get trainBaklavaEditor(): Editor {
    return this.train;
  }
  get modelBaklavaEditor(): Editor {
    return this.model;
  }
  private static instance: EditorManager;

  private model: Editor = new Editor();
  private train: Editor = new Editor();
  private data: Editor = new Editor();
  private overview: Editor = new Editor();

  private constructor() {}

  public static getInstance(): EditorManager {
    if (!EditorManager.instance) {
      EditorManager.instance = new EditorManager();
    }
    return EditorManager.instance;
  }

  public addNode(type: string, editor: EditorType): void {
    let whichEditor: Editor;
    switch (editor) {
      case EditorType.MODEL:
        whichEditor = this.model;
        break;
      case EditorType.DATA:
        whichEditor = this.data;
        break;
      case EditorType.TRAIN:
        whichEditor = this.train;
        break;
      case EditorType.OVERVIEW:
        whichEditor = this.overview;
        break;
      default:
        whichEditor = this.model;
    }
    const NodeType = whichEditor.nodeTypes.get(type);
    if (NodeType === undefined) {
      console.error(`Undefined Node Type: ${type}`);
    } else {
      whichEditor.addNode(new NodeType() as any);
    }
  }
}
