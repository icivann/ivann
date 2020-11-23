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
        <LayerTab/>
      </Tab>
      <Tab name="Custom">
        <CustomTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.DATA">
      <Tab name="Data Components">
        <DataTab/>
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
import ComponentsTab from '@/components/tabs/nodes/ComponentsTab.vue';
import LayerTab from '@/components/tabs/nodes/LayerTab.vue';
import DataTab from '@/components/tabs/nodes/DataTab.vue';

@Component({
  components: {
    DataTab,
    LayerTab,
    ComponentsTab,
    CustomTab,
    NodesTab,
    Tab,
    Tabs,
  },
  computed: mapGetters(['currEditorType']),
})
export default class Sidebar extends Vue {
  private editorType = EditorType;
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
