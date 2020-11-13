<template>
  <div>
    <ExpandablePanel :name="ioLabel">
      <ButtonGrid>
        <AddNodeButton :node="ioNodes.nodes[0].name" name="Input" :names="editorIONames"/>
        <AddNodeButton :node="ioNodes.nodes[1].name" name="Output" :names="editorIONames"/>
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel v-for="(category) in modelNodes.slice(1)" :key="category.category"
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
import { mapGetters } from 'vuex';
import EditorManager from '@/EditorManager';
import { ModelCategories } from '@/nodes/model/Types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['editorIONames']),
})
export default class LayersTab extends Vue {
  private modelNodes = EditorManager.getInstance().modelCanvas.nodeList;
  private ioNodes = this.modelNodes[0];
  private ioLabel = ModelCategories.IO;
}
</script>
