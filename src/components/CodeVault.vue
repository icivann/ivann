<template>
  <div class="vault">
    <Tabs full-screen-tab="IDE">
      <Tab name="Functions" :padded="false">
        <FunctionsTab/>
      </Tab>
      <Tab
        v-for="(file) of openFiles"
        :key="file.filename"
        :name="file.filename"
        :padded="false">
        <IdeTab :filename="file.filename"/>
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
import { Getter } from 'vuex-class';
import { ParsedFile } from '@/store/codeVault/types';

@Component({
  components: {
    FunctionsTab,
    IdeTab,
    Tab,
    Tabs,
  },
})
export default class CodeVault extends Vue {
  @Getter('openFiles') openFiles!: ParsedFile[];
}
</script>

<style scoped>
  .vault {
    color: var(--foreground);
    background: var(--dark-grey);
    height: calc(100vh - 2.5rem);
  }
</style>
