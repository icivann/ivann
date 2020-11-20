<template>
  <div>
    <SearchBar class="mb-2" @value-change="searchString"/>
    <ExpandablePanel v-for="(category) in displayNodes" :key="category.category"
                     :name="category.category" :inital-open="searching">
      <ButtonGrid>
        <AddNodeButton
          v-for="(node) in category.nodes" :key="node.name"
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
import { Component, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import EditorManager from '@/EditorManager';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';
import SearchBar from '@/components/SearchBar.vue';
import {
  convertToSearch, modify, search, SearchItem,
} from '@/components/SearchUtils';
import { Getter } from 'vuex-class';

@Component({
  components: {
    SearchBar,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class LayersTab extends Vue {
  private modelNodes = convertToSearch(EditorManager.getInstance().modelCanvas.nodeList);
  private displayNodes: SearchItem[] = [];
  private searching = false;
  @Getter('editorIONames') editorIONames!: Set<string>;

  created() {
    this.modelNodes = modify(this.modelNodes, ModelCategories.IO, ModelNodes.InModel, {
      name: ModelNodes.InModel,
      displayName: 'Input',
      names: this.editorIONames,
    });
    this.modelNodes = modify(this.modelNodes, ModelCategories.IO, ModelNodes.OutModel, {
      name: ModelNodes.OutModel,
      displayName: 'Output',
      names: this.editorIONames,
    });

    this.displayNodes = this.modelNodes;
  }

  private searchString(searchString: string) {
    if (searchString === '') {
      this.displayNodes = this.modelNodes;
      this.searching = false;
    }
    this.displayNodes = search(this.modelNodes, searchString);
    this.searching = true;
  }
}
</script>
