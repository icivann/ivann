<template>
  <Scrollable>
    <Padded>
      <SearchBar @value-change="search"/>
      <p class="information-text" v-if="searchString === ''">
        We support all of the torch.nn layers from the PyTorch library.
        For more information, check out the
        <a href="https://pytorch.org/docs/stable/nn.html#" target="_blank">
          official PyTorch documentation
        </a>.
      </p>
      <ExpandablePanel
        :name="overviewCategories.Model"
        v-show="searchString === '' || renderedModelEditors.length > 0"
      >
        <div class="msg" v-show="modelEditors.length <= 1">No Models Created</div>
        <ButtonGrid v-show="modelEditors.length > 1">
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
        :name="modelCategories.IO"
        v-show="shouldRender('Input') || shouldRender('Output')"
      >
        <ButtonGrid>
          <AddNodeButton :node="modelNodeTypes.InModel" name="Input" :names="editorIONames"
                         v-if="shouldRender('Input')"/>
          <AddNodeButton :node="modelNodeTypes.OutModel" name="Output" :names="editorIONames"
                         v-if="shouldRender('Output')"/>
        </ButtonGrid>
      </ExpandablePanel>
      <ExpandablePanel v-for="(category) in renderedNodes" :key="category.category"
                       :name="category.category" v-show="category.nodes.length > 0">
        <ButtonGrid>
          <AddNodeButton v-for="(node) in category.nodes" :key="node.name"
                         :node="node.name"
                         :name="node.name"
          />
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
import { mapGetters } from 'vuex';
import EditorManager from '@/EditorManager';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';
import SearchBar from '@/SearchBar.vue';
import Padded from '@/components/wrappers/Padded.vue';
import Scrollable from '@/components/wrappers/Scrollable.vue';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import { Getter } from 'vuex-class';
import { EditorModel } from '@/store/editors/types';

@Component({
  components: {
    Scrollable,
    Padded,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
    SearchBar,
  },
  computed: mapGetters(['editorIONames']),
})
export default class LayersTab extends Vue {
  private nodeList = EditorManager.getInstance().modelCanvas.nodeList;
  private modelCategories = ModelCategories;
  private modelNodeTypes = ModelNodes;
  private searchString = '';
  private readonly overviewNodes = OverviewNodes;
  private readonly overviewCategories = OverviewCategories;
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];
  @Getter('currEditorModel') currEditorModel!: EditorModel;

  private get renderedNodes() {
    return this.nodeList.map((section) => ({
      category: section.category,
      nodes: section.nodes.filter((node) => this.shouldRender(node.name)),
    }));
  }

  private get renderedModelEditors() {
    const { name } = this.currEditorModel;
    return this.modelEditors
      .filter((editor) => this.shouldRender(editor.name) && name !== editor.name);
  }

  private search(search: string) {
    this.searchString = search;
  }

  private shouldRender(button: string) {
    return button.toLowerCase().includes(this.searchString.toLowerCase());
  }
}
</script>

<style lang="scss" scoped>
.information-text {
  font-weight: 200;
  font-size: 0.8rem;
}

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
