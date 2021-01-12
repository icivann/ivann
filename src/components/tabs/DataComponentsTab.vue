<template>
  <Scrollable>
    <Padded>
      <SearchBar @value-change="search"/>
      <ExpandablePanel
        :name="dataCategories.IO"
        v-show="shouldRender('Output')"
      >
        <ButtonGrid>
          <AddNodeButton :node="dataNodeTypes.OutData" name="Output"
                         v-if="shouldRender('Output')">
            <img src="@/assets/images/output-icon.svg" alt="Output"/>
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
import { mapGetters } from 'vuex';
import EditorManager from '@/EditorManager';
import { DataCategories, DataNodes } from '@/nodes/data/Types';
import SearchBar from '@/SearchBar.vue';
import Padded from '@/components/wrappers/Padded.vue';
import Scrollable from '@/components/wrappers/Scrollable.vue';

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
export default class DataComponentsTab extends Vue {
  private nodeList = EditorManager.getInstance().dataCanvas.nodeList;
  private dataCategories = DataCategories;
  private dataNodeTypes = DataNodes;
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
