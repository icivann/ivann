import ModelCanvas from '@/components/canvas/ModelCanvas';
import DataCanvas from '@/components/canvas/DataCanvas';
import TrainCanvas from '@/components/canvas/TrainCanvas';
import OverviewCanvas from '@/components/canvas/OverviewCanvas';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';
import Dropdown from '@/baklava/options/Dropdown.vue';
import Checkbox from '@/baklava/options/Checkbox.vue';
import CustomNode from '@/baklava/CustomNode.vue';
import TextArea from '@/baklava/options/TextArea.vue';
import { Engine } from '@baklavajs/plugin-engine';

export default class EditorManager {
  private static instance: EditorManager;

  private overview: OverviewCanvas = new OverviewCanvas();
  private model: ModelCanvas = new ModelCanvas();
  private data: DataCanvas = new DataCanvas();
  private train: TrainCanvas = new TrainCanvas();

  private view: ViewPlugin = new ViewPlugin();

  private eng: Engine = new Engine(true);

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

  get viewPlugin(): ViewPlugin {
    return this.view;
  }

  get engine(): Engine {
    return this.eng;
  }

  public resetView(): void {
    this.view.panning = { x: 0, y: 0 };
    this.view.scaling = 1;
  }

  private constructor() {
    this.view.registerOption('VectorOption', Vector);
    this.view.registerOption('IntOption', Integer);
    this.view.registerOption('DropdownOption', Dropdown);
    this.view.registerOption('TickBoxOption', Checkbox);
    this.view.registerOption('TextAreaOption', TextArea);

    this.view.components.node = CustomNode;
  }

  public static getInstance(): EditorManager {
    if (!EditorManager.instance) {
      EditorManager.instance = new EditorManager();
    }
    return EditorManager.instance;
  }
}
