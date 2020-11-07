<template>
  <div class="vault">
    <Tabs full-screen-tab="IDE">
      <Tab name="Functions">
        <FunctionsTab/>
      </Tab>
      <Tab class="h-100" name="IDE">
        <IdeTab/>
      </Tab>
    </Tabs>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import FunctionsTab from '@/components/tabs/FunctionsTab.vue';
import IdeTab from '@/components/tabs/IdeTab.vue';

@Component({
  components: {
    FunctionsTab,
    IdeTab,
    Tab,
    Tabs,
  },
})
export default class CodeVault extends Vue {
}
</script>

<style scoped>
  .vault {
    color: var(--foreground);
    background: var(--dark-grey);
    height: calc(100vh - 2.5rem);
  }
</style>
