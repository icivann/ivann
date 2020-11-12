<template>
  <div class="layers-tab">
    <ExpandablePanel v-for="(category) in modelNodes" :key="category.category"
                     :name="category.category">
      <ButtonGrid>
        <AddNodeButton v-for="(node) in category.nodes" :key="node.name"
                       :node="node.name"
                       :name="node.name"
                       :names="node.name === 'InModel' || node.name === 'OutModel'
                        ? editorIONames : undefined"
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
}
</script>
