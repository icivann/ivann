<template>
  <div class="vault">
    <Tabs
      :selected-tab-index="tabIndex"
      @changeTab="switchTab"
    >
      <Tab name="Functions" :padded="false">
        <FunctionsTab/>
      </Tab>
      <Tab
        v-for="(file) of openFiles"
        :key="file.filename"
        :name="file.filename"
        :padded="false"
      >
        <IdeTab :filename="file.filename" @closeTab="close"/>
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
import { mapGetters } from 'vuex';

@Component({
  components: {
    FunctionsTab,
    IdeTab,
    Tab,
    Tabs,
  },
  computed: mapGetters(['openFiles']),
})
export default class CodeVault extends Vue {
  private tabIndex = 0;

  private switchTab(newTab: number): void {
    this.tabIndex = newTab;
  }

  private close(): void {
    this.tabIndex -= 1;
  }
}
</script>

<style scoped>
  .vault {
    color: var(--foreground);
    background: var(--dark-grey);
    height: calc(100vh - 2.5rem);
  }
</style>
