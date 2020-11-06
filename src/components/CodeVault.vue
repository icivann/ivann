<template>
  <div class="vault">
    <Tabs>
      <Tab name="Functions">
        <FunctionsTab/>
      </Tab>
      <Tab name="Notepad">
        Notepad stuff goes here
      </Tab>
    </Tabs>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import FunctionsTab from '@/components/tabs/FunctionsTab.vue';

@Component({
  components: {
    FunctionsTab,
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
