<template>
  <div class="right">
    <Tabs v-show="currEditorType === editorType.OVERVIEW">
      <Tab name="Components">
        <ComponentsTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.MODEL">
      <Tab name="Layers">
        <LayersTab/>
      </Tab>
      <Tab name="Search">
        <SearchTab/>
      </Tab>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.DATA">
      <Tab name="Data Stuff"/>
    </Tabs>
    <Tabs v-show="currEditorType === editorType.TRAIN">
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
