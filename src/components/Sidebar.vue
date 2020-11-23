<template>
  <div class="right">
    <Tabs v-show="currEditorType === editorType.OVERVIEW">
      <Tab name="Components">
        <ComponentsTab/>
      </Tab>
      <Tab name="Custom">
        <CustomTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.MODEL">
      <Tab name="Layers">
        <NodesTab :searchItems="layersTab.searchItems"/>
      </Tab>
      <Tab name="Custom">
        <CustomTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.DATA">
      <Tab name="Data Components">
        <NodesTab :searchItems="dataComponentsTab.searchItems"/>
      </Tab>
      <Tab name="Custom">
        <CustomTab/>
      </Tab>
    </Tabs>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import CustomTab from '@/components/tabs/nodes/CustomTab.vue';
import EditorType from '@/EditorType';
import { mapGetters } from 'vuex';
import NodesTab from '@/components/tabs/nodes/NodesTab.vue';
import LayersTab from '@/components/tabs/nodes/LayersTab';
import { Getter } from 'vuex-class';
import DataComponentsTab from '@/components/tabs/nodes/DataComponentsTab';
import ComponentsTab from '@/components/tabs/nodes/ComponentsTab.vue';

@Component({
  components: {
    ComponentsTab,
    CustomTab,
    NodesTab,
    Tab,
    Tabs,
  },
  computed: mapGetters(['currEditorType']),
})
export default class Sidebar extends Vue {
  @Getter('editorIONames') editorIONames!: Set<string>;
  private editorType = EditorType;
  private layersTab!: LayersTab;
  private dataComponentsTab!: DataComponentsTab;

  created() {
    /* Initialise here, otherwise getters are called after constructors. */
    this.layersTab = new LayersTab(this.editorIONames);
    this.dataComponentsTab = new DataComponentsTab(this.editorIONames);
  }
}
</script>

<style scoped>
  .right {
    background: var(--background-alt);
    color: var(--foreground);
    border-left: 1px solid var(--grey);
    height: calc(100vh - 2.5rem);
    width: 100%;
  }
</style>
