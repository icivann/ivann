<template>
  <NodesTab :searchItems="componentsTab.searchItems"/>
</template>

<script lang="ts">
import {
  Component, Vue, Watch,
} from 'vue-property-decorator';
import { EditorModel } from '@/store/editors/types';
import { Getter } from 'vuex-class';
import ComponentsNodesTab from '@/components/tabs/nodes/ComponentsNodesTab';
import NodesTab from './NodesTab.vue';

@Component({
  components: { NodesTab },
})
export default class ComponentsTab extends Vue {
  private componentsTab!: ComponentsNodesTab;
  @Getter('modelEditors') modelEditors!: EditorModel[];
  @Getter('dataEditors') dataEditors!: EditorModel[];

  created() {
    this.componentsTab = new ComponentsNodesTab(this.modelEditors, this.dataEditors);
  }

  @Watch('modelEditors')
  private updateModelEditors(newModels: EditorModel[]) {
    this.componentsTab.updateModelEditors(newModels);
  }

  @Watch('dataEditors')
  private updateDataEditors(newModels: EditorModel[]) {
    this.componentsTab.updateDataEditors(newModels);
  }
}
</script>
