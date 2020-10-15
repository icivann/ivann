<template>
  <div class="right-sidebar h-100">
    <Tabs>
      <Tab name="Layers">Yo</Tab>
      <Tab name="Search">Yo again!</Tab>
    </Tabs>
  </div>
</template>

<script>
import Tabs from '@/components/Tabs.vue';
import Tab from '@/components/Tab.vue';

export default {
  components: { Tab, Tabs },
};
</script>

<style scoped>
  .right-sidebar {
    background: var(--background-alt);
    color: var(--foreground);
    border-left: 0.125vw solid var(--foreground);
  }
</style>
