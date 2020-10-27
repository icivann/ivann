<template>
  <div class="right">
    <Tabs>
      <Tab :name="nodesTab">
        <LayersTab v-if="isModels"/>
        <ComponentsTab v-if="isOverview"/>
      </Tab>
      <Tab name="Search">
        <SearchTab/>
      </Tab>
    </Tabs>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import LayersTab from '@/components/tabs/LayersTab.vue';
import SearchTab from '@/components/tabs/SearchTab.vue';
import EditorType from '@/EditorType';
import ComponentsTab from '@/components/tabs/ComponentsTab.vue';

@Component({
  components: {
    ComponentsTab,
    SearchTab,
    LayersTab,
    Tab,
    Tabs,
  },
})
export default class Sidebar extends Vue {
  private isModels = true;
  private isOverview = false;
  private isTrain = false;
  private isData = false;
  private nodesTab = 'Layers'

  @Watch('$store.getters.currEditorType')
  onEditorsUpdate(editor: EditorType) {
    this.isModels = false;
    this.isOverview = false;
    this.isTrain = false;
    this.isData = false;

    switch (editor) {
      case EditorType.OVERVIEW:
        this.nodesTab = 'Components';
        this.isOverview = true;
        break;
      case EditorType.MODEL:
        this.nodesTab = 'Layers';
        this.isModels = true;
        break;
      case EditorType.DATA:
        this.nodesTab = 'DataMagic';
        this.isData = true;
        break;
      case EditorType.TRAIN:
        this.nodesTab = 'SteamTrain';
        this.isTrain = true;
        break;
      default:
        this.nodesTab = 'UNKOWN';
        break;
    }
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
