import ModelCanvas from '@/components/canvas/ModelCanvas';
import DataCanvas from '@/components/canvas/DataCanvas';
import OverviewCanvas from '@/components/canvas/OverviewCanvas';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';
import Dropdown from '@/baklava/options/Dropdown.vue';
import Checkbox from '@/baklava/options/Checkbox.vue';
import { Engine } from '@baklavajs/plugin-engine';
import CustomNode from '@/baklava/CustomNode.vue';
import TextArea from '@/baklava/options/TextArea.vue';
import CodeVaultButton from '@/baklava/options/CodeVaultButton.vue';
import CustomContextMenu from '@/baklava/CustomContextMenu.vue';

export default class EditorManager {
  private static instance: EditorManager;

  private overview: OverviewCanvas = new OverviewCanvas();
  private model: ModelCanvas = new ModelCanvas();
  private data: DataCanvas = new DataCanvas();

  private view: ViewPlugin = new ViewPlugin();

  private eng: Engine = new Engine(true);

  private dropStatus = false;

  get overviewCanvas(): OverviewCanvas {
    return this.overview;
  }
  get modelCanvas(): ModelCanvas {
    return this.model;
  }
  get dataCanvas(): DataCanvas {
    return this.data;
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

  public enableDrop(value: boolean): void {
    this.dropStatus = value;
  }

  get canDrop(): boolean {
    return this.dropStatus;
  }

  private constructor() {
    this.view.registerOption('VectorOption', Vector);
    this.view.registerOption('IntOption', Integer);
    this.view.registerOption('DropdownOption', Dropdown);
    this.view.registerOption('TickBoxOption', Checkbox);
    this.view.registerOption('TextAreaOption', TextArea);
    this.view.registerOption('CodeVaultButtonOption', CodeVaultButton);

    this.view.components.node = CustomNode;
    this.view.components.contextMenu = CustomContextMenu;
  }

  public static getInstance(): EditorManager {
    if (!EditorManager.instance) {
      EditorManager.instance = new EditorManager();
    }
    return EditorManager.instance;
  }
}
