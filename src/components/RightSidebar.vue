<template>
  <div class="right-sidebar h-100">
    <Tabs>
      <Tab name="Layers">On this tab, you can create Layer Nodes.</Tab>
      <Tab name="Search">This tab is for searching.</Tab>
    </Tabs>
  </div>
</template>

<script>
import Tabs from '@/components/Tabs.vue';
import Tab from '@/components/Tab.vue';

export default {
  components: {
    Tab,
    Tabs,
  },
};
</script>

<style scoped>
  .right-sidebar {
    background: var(--background-alt);
    color: var(--foreground);
    border-left: 0.125vw solid var(--grey);
  }
</style>
