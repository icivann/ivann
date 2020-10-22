import ModelCanvas from '@/components/canvas/ModelCanvas';
import DataCanvas from '@/components/canvas/DataCanvas';
import TrainCanvas from '@/components/canvas/TrainCanvas';
import OverviewCanvas from '@/components/canvas/OverviewCanvas';

export default class EditorManager {
  private static instance: EditorManager;

  private overview: OverviewCanvas = new OverviewCanvas();
  private model: ModelCanvas = new ModelCanvas();
  private data: DataCanvas = new DataCanvas();
  private train: TrainCanvas = new TrainCanvas();

  get overviewCanvas(): OverviewCanvas {
    return this.overview;
  }
  get modelCanvas(): ModelCanvas {
    return this.model;
  }
  get dataCanvas(): DataCanvas {
    return this.data;
  }
  get trainCanvas(): TrainCanvas {
    return this.train;
  }

  private constructor() {
  }

  public static getInstance(): EditorManager {
    if (!EditorManager.instance) {
      EditorManager.instance = new EditorManager();
    }
    return EditorManager.instance;
  }
}
