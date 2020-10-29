<template>
  <div>
    <ExpandablePanel name="Models">
      <ButtonGrid>
        <AddNodeButton v-for="editor in modelEditors"
                       :node="modelNodeType"
                       :options="editor"
                       :key="editor.name"
                       :name="editor.name"/>
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
import { Overview } from '@/nodes/model/Types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['modelEditors']),
})
export default class ComponentsTab extends Vue {
  private modelNodeType = Overview.ModelNode;
}
</script>
