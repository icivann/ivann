<template>
  <div class="right">
    <Tabs v-show="currEditorType === modelType">
      <Tab name="Layers">
        <LayersTab/>
      </Tab>
      <Tab name="Search">
        <SearchTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === overviewType">
      <Tab name="Components">
        <ComponentsTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === dataType">
      <Tab name="Data Stuff"/>
    </Tabs>
    <Tabs v-show="currEditorType === trainType">
      <Tab name="Steam Training"/>
    </Tabs>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import LayersTab from '@/components/tabs/LayersTab.vue';
import SearchTab from '@/components/tabs/SearchTab.vue';
import EditorType from '@/EditorType';
import ComponentsTab from '@/components/tabs/ComponentsTab.vue';
import { mapGetters } from 'vuex';

@Component({
  components: {
    ComponentsTab,
    SearchTab,
    LayersTab,
    Tab,
    Tabs,
  },
  computed: mapGetters(['currEditorType']),
})
export default class Sidebar extends Vue {
  private modelType = EditorType.MODEL;
  private dataType = EditorType.DATA;
  private trainType = EditorType.TRAIN;
  private overviewType = EditorType.OVERVIEW;
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
