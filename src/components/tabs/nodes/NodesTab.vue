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
          :options="node.options"
          :names="node.names"
        />
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import {
  Component, Prop, Vue, Watch,
} from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import SearchBar from '@/components/SearchBar.vue';
import { search, SearchItem } from '@/components/SearchUtils';
import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';

@Component({
  components: {
    SearchBar,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class NodesTab extends Vue {
  @Prop({ required: true }) searchItems!: SearchItem[];
  private enabledNodes: SearchItem[] = [];
  private renderedNodes: SearchItem[] = [];

  @Watch('searchItems', { deep: true })
  private changeNodesTab(newItems: SearchItem[]) {
    this.enabledNodes = newItems;
  }

  created() {
    this.enabledNodes = this.searchItems;
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
