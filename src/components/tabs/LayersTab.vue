<template>
  <div>
    <SearchBar @value-change="search"/>
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
  </div>
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

@Component({
  components: {
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

  private get renderedNodes() {
    return this.nodeList.map((section) => ({
      category: section.category,
      nodes: section.nodes.filter((node) => this.shouldRender(node.name)),
    }));
  }

  private search(search: string) {
    this.searchString = search;
  }

  private shouldRender(button: string) {
    return button.toLowerCase().includes(this.searchString.toLowerCase());
  }
}
</script>
