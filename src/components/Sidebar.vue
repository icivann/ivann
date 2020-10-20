<template>
  <div class="right">
    <Tabs>
      <Tab name="Layers"><LayersTab/></Tab>
      <Tab name="Search"><SearchTab/></Tab>
    </Tabs>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import LayersTab from '@/components/tabs/LayersTab.vue';
import SearchTab from '@/components/tabs/SearchTab.vue';

@Component({
  components: {
    SearchTab,
    LayersTab,
    Tab,
    Tabs,
  },
})
export default class Sidebar extends Vue {
}
</script>

<style scoped>
  .right {
    background: var(--background-alt);
    color: var(--foreground);
    border-left: 0.125vw solid var(--grey);
    height: calc(100vh - 2.5rem);
    width: 100%;
  }
</style>
