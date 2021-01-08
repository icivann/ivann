<template>
  <div class="right">
    <Tabs v-show="currEditorType === editorType.OVERVIEW" data-v-step="sidebar-overview">
      <Tab name="Components">
        <ComponentsTab/>
      </Tab>
      <Tab name="Custom">
        <CustomTab :custom-node="overviewCustomNode" />
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.MODEL" data-v-step="sidebar-model">
      <Tab name="Layers">
        <LayersTab/>
      </Tab>
      <Tab name="Custom">
        <CustomTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.DATA" data-v-step="sidebar-data">
      <Tab name="Data Components">
        <DataComponentsTab/>
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
import LayersTab from '@/components/tabs/LayersTab.vue';
import CustomTab from '@/components/tabs/CustomTab.vue';
import EditorType from '@/EditorType';
import ComponentsTab from '@/components/tabs/ComponentsTab.vue';
import { mapGetters } from 'vuex';
import DataComponentsTab from '@/components/tabs/DataComponentsTab.vue';
import { OverviewNodes } from '@/nodes/overview/Types';

@Component({
  components: {
    DataComponentsTab,
    ComponentsTab,
    CustomTab,
    LayersTab,
    Tab,
    Tabs,
  },
  computed: mapGetters(['currEditorType']),
})
export default class Sidebar extends Vue {
  private editorType = EditorType;
  private overviewCustomNode = OverviewNodes.Custom;
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
