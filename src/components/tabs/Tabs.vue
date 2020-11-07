<template>
  <div class="h-100 all">
    <div class="tabs">
      <div class="tab-head" :class="index === selected && 'selected'"
           v-for="(tab, index) in tabs" :key="index"
           @click="selectTab(index, tab.name)">
        {{tab.name}}
      </div>
    </div>
    <div class="tab-content" v-bind:class="{ 'fill-parent': fullScreenTab === selectedName }">
      <slot/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import Tab from '@/components/tabs/Tab.vue';

@Component
export default class Tabs extends Vue {
  @Prop({ default: undefined }) readonly fullScreenTab?: string;
  private tabs = this.$children as [Tab];
  private selected = 0;
  private selectedName = '';

  private selectTab(given: number, tabName: string) {
    this.selected = given;
    this.selectedName = tabName;
    this.tabs.forEach((tab, index) => {
      tab.setVisible(index === given);
    });
  }

  mounted() {
    this.tabs[0].setVisible(true);
    this.selectedName = this.tabs[0].name;
  }
}
</script>

<style scoped>
  .tabs {
    display: flex;
    height: 60px;
    background: var(--background);
    overflow-x: hidden;
  }

  .tab-head {
    margin: 0.5em 1em 0;
    padding: 0.2em 1em;
    border-bottom-style: none;
    border-bottom: var(--blue);
    user-select: none;
    transition-duration: 0.1s;
  }

  .tab-head.selected {
    border-bottom-width: 4px;
    border-bottom-style: solid;
    transition-duration: 0.1s;
  }

  .tab-head:hover {
    cursor: pointer;
    border-bottom-width: 1px;
    border-bottom-style: solid;
  }

  .tab-head.selected:hover {
    border-bottom-width: 4px;
  }

  .tab-content {
    margin: 1em;
    max-height: calc(100% - 60px - 2em);
    overflow: auto;
    scrollbar-width: none;
  }

  .tab-content.fill-parent {
    margin: 0;
    height: 100%;
    max-height: calc(100% - 60px);
    overflow: auto;
    scrollbar-width: none;
  }

  ::-webkit-scrollbar {
    width: 0;
    background: none;
  }
</style>
