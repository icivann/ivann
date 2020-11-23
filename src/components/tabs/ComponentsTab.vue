<template>
  <div>
    <SearchBar class="mb-2" @value-change="searchString"/>
    <ExpandablePanel v-for="(category) in renderedNodes" :key="category.category"
                     :name="category.category">
      <ButtonGrid>
        <AddNodeButton
          v-for="(node) in category.nodes" :key="node.displayName"
          :node="node.name"
          :name="node.displayName"
          :names="node.names"
          :options="node.options"
        />
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import SearchBar from '@/components/SearchBar.vue';
import { search, SearchItem } from '@/components/SearchUtils';
import { EditorModel } from '@/store/editors/types';
import { Getter } from 'vuex-class';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
    SearchBar,
  },
})
export default class ComponentsTab extends Vue {
  private enabledNodes: SearchItem[] = [];
  private renderedNodes: SearchItem[] = [];
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];

  @Watch('modelEditors')
  private updateModelEditors(newModels: EditorModel[]) {
    this.enabledNodes[0].nodes = newModels
      .map((model) => ({
        name: OverviewNodes.ModelNode,
        displayName: model.name,
        options: model,
      }));
  }

  @Watch('dataEditors')
  private updateDataEditors(newModels: EditorModel[]) {
    this.enabledNodes[1].nodes = newModels
      .map((model) => ({
        name: OverviewNodes.ModelNode,
        displayName: model.name,
        options: model,
      }));
  }

  created() {
    this.enabledNodes = [
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
    this.renderedNodes = this.enabledNodes;
  }

  private searchString(searchString: string) {
    if (searchString === '') {
      this.renderedNodes = this.enabledNodes;
      return;
    }
    this.renderedNodes = search(this.enabledNodes, searchString);
  }
}
</script>

<style scoped>
  .msg {
    text-align: center;
    background: var(--background);
    border-radius: 4px;
    border-style: solid;
    border-width: 1px;
    margin-top: 5px;
    border-color: var(--grey);
    font-size: smaller;
  }
</style>
