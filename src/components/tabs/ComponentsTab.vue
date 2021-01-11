<template>
  <Scrollable>
    <Padded>
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
            :overviewFlag="true"
            :key="editor.name"
            :name="editor.name">
            <img src="@/assets/images/nn_logo.svg" :alt="editor.name"/>
          </AddNodeButton>
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
            :name="editor.name">
            <img src="@/assets/images/data-icon.svg" :alt="editor.name"/>
          </AddNodeButton>
        </ButtonGrid>
      </ExpandablePanel>
      <ExpandablePanel v-for="(category) in renderedNodes" :key="category.category"
                       :name="category.category" v-show="category.nodes.length > 0">
        <ButtonGrid>
          <AddNodeButton v-for="(node) in category.nodes" :key="node.name"
                         :node="node.name"
                         :name="node.name"
          >
            <img v-if="node.img !== undefined" :alt="node.name"
                 :src="require(`@/assets/images/${node.img}`)"/>
          </AddNodeButton>
        </ButtonGrid>
      </ExpandablePanel>
    </Padded>
  </Scrollable>
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
import EditorManager from '@/EditorManager';
import Scrollable from '@/components/wrappers/Scrollable.vue';
import Padded from '@/components/wrappers/Padded.vue';

@Component({
  components: {
    Padded,
    Scrollable,
    SearchBar,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class ComponentsTab extends Vue {
  private readonly overviewNodes = OverviewNodes;
  private readonly overviewCategories = OverviewCategories;
  private nodeList = EditorManager.getInstance().overviewCanvas.nodeList;
  private searchString = '';
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];

  private search(search: string) {
    this.searchString = search;
  }

  private get renderedModelEditors() {
    return this.modelEditors
      .filter((editor) => this.shouldRender(editor.name));
  }

  private get renderedDataEditors() {
    return this.dataEditors
      .filter((editor) => this.shouldRender(editor.name));
  }

  private get renderedNodes() {
    return this.nodeList.map((section) => ({
      category: section.category,
      nodes: section.nodes.filter((node) => this.shouldRender(node.name)),
    }));
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
