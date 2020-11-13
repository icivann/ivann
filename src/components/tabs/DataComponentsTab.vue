<template>
  <div>
    <ExpandablePanel :name="ioLabel">
      <ButtonGrid>
        <AddNodeButton :node="ioNodes.nodes[0].name" name="Input" :names="editorIONames"/>
        <AddNodeButton :node="ioNodes.nodes[1].name" name="Output"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel v-for="(category) in dataNodes.slice(1)" :key="category.category"
                     :name="category.category">
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
import EditorManager from '@/EditorManager';
import { DataCategories } from '@/nodes/data/Types';
import { mapGetters } from 'vuex';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['editorIONames']),
})
export default class DataComponentsTab extends Vue {
  private dataNodes = EditorManager.getInstance().dataCanvas.nodeList;
  private ioNodes = this.dataNodes[0];
  private ioLabel = DataCategories.IO;
}
</script>
