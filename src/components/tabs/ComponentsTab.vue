<template>
  <div>
    <SearchBar @value-change="search"/>
    <ExpandablePanel
      :name="overviewCategories.Model"
      v-show="searchString === '' || renderedModelEditors.length > 0"
    >
      <div class="msg" v-show="modelEditors.length === 0">No Models Created</div>
      <ButtonGrid v-show="modelEditors.length > 0">
        <AddNodeButton
          v-for="editor in renderedModelEditors"
          :node="overviewNodes.ModelNode"
          :options="editor"
          :key="editor.name"
          :name="editor.name"
        />
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel
      :name="overviewCategories.Data"
      v-show="searchString === '' || renderedDataEditors.length > 0"
    >
      <div class="msg" v-show="dataEditors.length === 0">No Datasets Created</div>
      <ButtonGrid v-show="dataEditors.length > 0">
        <AddNodeButton
          v-for="editor in renderedDataEditors"
          :node="overviewNodes.DataNode"
          :options="editor"
          :key="editor.name"
          :name="editor.name"
        />
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel
      :name="overviewCategories.Train"
      v-show="shouldRender('Train Classifier')"
    >
      <ButtonGrid>
        <AddNodeButton :node="overviewNodes.TrainClassifier" name="Train Classifier"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel
      :name="overviewCategories.Optimizer"
      v-show="shouldRender(overviewNodes.Adadelta)"
    >
      <ButtonGrid>
        <AddNodeButton :node="overviewNodes.Adadelta" :name="overviewNodes.Adadelta"/>
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import SearchBar from '@/SearchBar.vue';
import { Getter } from 'vuex-class';
import { EditorModel } from '@/store/editors/types';

@Component({
  components: {
    SearchBar,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class ComponentsTab extends Vue {
  private readonly overviewNodes = OverviewNodes;
  private readonly overviewCategories = OverviewCategories;
  private searchString = '';
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];

  private search(search: string) {
    this.searchString = search;
  }

  private get renderedModelEditors() {
    return this.modelEditors
      .filter((editor) => editor.name.toLowerCase().includes(this.searchString.toLowerCase()));
  }

  private get renderedDataEditors() {
    return this.dataEditors
      .filter((editor) => this.shouldRender(editor.name));
  }

  private shouldRender(button: string) {
    return button.toLowerCase().includes(this.searchString.toLowerCase());
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
