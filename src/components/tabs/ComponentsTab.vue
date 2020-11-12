<template>
  <div>
    <ExpandablePanel name="Models">
      <ButtonGrid>
        <AddNodeButton
          v-for="editor in modelEditors"
          :node="overviewNodes.ModelNode"
          :options="editor"
          :key="editor.name"
          :name="editor.name"
        />
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Datasets">
      <ButtonGrid>
        <AddNodeButton
          v-for="editor in dataEditors"
          :node="overviewNodes.DataNode"
          :options="editor"
          :key="editor.name"
          :name="editor.name"
        />
      </ButtonGrid>
    </ExpandablePanel>
    <ExpandablePanel name="Custom">
      <ButtonGrid>
        <AddNodeButton node="Custom" name="Custom"/>
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
import { OverviewNodes } from '@/nodes/overview/Types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['modelEditors', 'dataEditors']),
})
export default class ComponentsTab extends Vue {
  private overviewNodes = OverviewNodes;
}
</script>
