<template>
  <NodesTab :searchItems="searchItems"/>
</template>

<script lang="ts">
import {
  Component, Vue, Watch,
} from 'vue-property-decorator';
import { EditorModel } from '@/store/editors/types';
import { Getter } from 'vuex-class';
import { SearchItem } from '@/components/SearchUtils';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import NodesTab from './NodesTab.vue';

@Component({
  components: { NodesTab },
})
export default class ComponentsTab extends Vue {
  private searchItems!: SearchItem[];
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];

  created() {
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
    this.updateModelEditors(this.modelEditors);
    this.updateDataEditors(this.dataEditors);
  }

  @Watch('modelEditors')
  private updateModelEditors(newModels: EditorModel[]) {
    this.updateEditors(newModels, 0, OverviewNodes.ModelNode);
  }

  @Watch('dataEditors')
  private updateDataEditors(newModels: EditorModel[]) {
    this.updateEditors(newModels, 1, OverviewNodes.DataNode);
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
</script>
