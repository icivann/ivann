<template>
  <div class="h-100 all">
    <div class="tabs">
      <div class="tab-head" :class="index === selected && 'selected'"
           v-for="(tab, index) in tabs" :key="index"
           @click="selectTab(index)">
        {{tab.name}}
      </div>
    </div>
    <div class="tab-content">
      <slot/>
    </div>
  </div>
</template>

<script lang="ts">
import {
  Component, Prop, Vue, Watch,
} from 'vue-property-decorator';
import Tab from '@/components/tabs/Tab.vue';

@Component
export default class Tabs extends Vue {
  @Prop() selectedTabIndex?: number;
  private tabs = this.$children as [Tab];
  private selected = 0;

  @Watch('selectedTabIndex')
  private onTabSelectChange(newVal: number, oldVal: number) {
    if (newVal !== oldVal) {
      this.tabs[newVal].setVisible(true);
      this.selected = newVal;
      if (oldVal < this.tabs.length) this.tabs[oldVal].setVisible(false);
    }
  }

  private selectTab(given: number) {
    if (this.selectedTabIndex === undefined) {
      this.selected = given;
      this.tabs.forEach((tab, index) => {
        tab.setVisible(index === given);
      });
    } else {
      this.$emit('changeTab', given);
    }
  }

  mounted() {
    this.tabs[0].setVisible(true);
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
    border-bottom: solid var(--background) 1px;
    user-select: none;
    transition-duration: 0.1s;
  }

  .tab-head.selected {
    border-bottom-width: 4px;
    transition-duration: 0.1s;
    border-bottom-color: var(--blue);
  }

  .tab-head:hover {
    border-bottom-width: 1px;
    border-bottom-color: var(--blue);
    transition-duration: 0.1s;
    cursor: pointer;
  }

  .tab-head.selected:hover {
    transition-duration: 0.1s;
    border-bottom-color: #1B67E0;
    border-bottom-width: 4px;
  }

  .tab-content {
    height: 100%;
    max-height: calc(100% - 60px);
  }
</style>
