import { Editor } from '@baklavajs/core';
import EditorType from '@/EditorType';
import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import ModelCanvas from '@/components/canvas/ModelCanvas';
import DataCanvas from '@/components/canvas/DataCanvas';
import TrainCanvas from '@/components/canvas/TrainCanvas';
import OverviewCanvas from '@/components/canvas/OverviewCanvas';

export default class EditorManager {
  private static instance: EditorManager;

  private model: Editor = new Editor();
  private train: Editor = new Editor();
  private data: Editor = new Editor();
  private overview: Editor = new Editor();

  private modelAbstractCanvas: AbstractCanvas = new ModelCanvas();
  private dataAbstractCanvas: AbstractCanvas = new DataCanvas();
  private trainAbstractCanvas: AbstractCanvas = new TrainCanvas();
  private overviewAbstractCanvas: AbstractCanvas = new OverviewCanvas();

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

  get modelCanvas(): AbstractCanvas {
    return this.modelAbstractCanvas;
  }

  get dataCanvas(): AbstractCanvas {
    return this.dataAbstractCanvas;
  }

  get trainCanvas(): AbstractCanvas {
    return this.trainAbstractCanvas;
  }

  get overviewCanvas(): AbstractCanvas {
    return this.overviewAbstractCanvas;
  }

  private constructor() {
  }

  public static getInstance(): EditorManager {
    if (!EditorManager.instance) {
      EditorManager.instance = new EditorManager();
    }
    return EditorManager.instance;
  }

  public addNode(type: string, editor: number): void {
    let whichEditor: Editor | undefined;
    switch (editor) {
      case 0:
        whichEditor = this.model;
        break;
      case 1:
        whichEditor = this.data;
        break;
      case 2:
        whichEditor = this.train;
        break;
      case 3:
        whichEditor = this.overview;
        break;
      default:
        break;
    }

    if (whichEditor === undefined) {
      console.error(`Editor state ${editor} does not exist`);
      return;
    }

    const NodeType = whichEditor.nodeTypes.get(type);
    if (NodeType === undefined) {
      console.error(`Undefined Node Type: ${type}`);
    } else {
      whichEditor.addNode(new NodeType() as any);
    }
  }
}
