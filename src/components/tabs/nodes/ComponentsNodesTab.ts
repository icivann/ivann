import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { SearchItem } from '@/components/SearchUtils';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import { EditorModel } from '@/store/editors/types';

export default class ComponentsNodesTab extends AbstractNodesTab {
  private static readonly MODEL_CATEGORY = 0;
  private static readonly DATA_CATEGORY = 1;

  public readonly searchItems: SearchItem[];

  public constructor(modelEditors: EditorModel[], dataEditors: EditorModel[]) {
    super();
    this.searchItems = [
      { category: OverviewCategories.Model, nodes: [] },
      { category: OverviewCategories.Data, nodes: [] },
      {
        category: OverviewCategories.Train,
        nodes: [{
          name: OverviewNodes.TrainClassifier,
          displayName: 'Train Classifier',
        }],
      },
      {
        category: OverviewCategories.Optimizer,
        nodes: [{
          name: OverviewNodes.Adadelta,
          displayName: OverviewNodes.Adadelta,
        }],
      },
    ];
    this.updateModelEditors(modelEditors);
    this.updateDataEditors(dataEditors);
  }

  public updateModelEditors(modelEditors: EditorModel[]): void {
    this.updateEditors(modelEditors, ComponentsNodesTab.MODEL_CATEGORY, OverviewNodes.ModelNode);
  }

  public updateDataEditors(dataEditors: EditorModel[]): void {
    this.updateEditors(dataEditors, ComponentsNodesTab.DATA_CATEGORY, OverviewNodes.DataNode);
  }

  private updateEditors(editors: EditorModel[], category: number, type: string): void {
    this.searchItems[category].nodes = editors
      .map((model) => ({
        name: type,
        displayName: model.name,
        options: model,
      }));
  }
}
