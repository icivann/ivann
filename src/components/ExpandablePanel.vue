<template>
  <div class="panel">
    <div class="header" @click="toggleOpen">
      {{name}}
    </div>
    <slot v-if="open"/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class ExpandablePanel extends Vue {
  @Prop({ required: true }) readonly name!: string

  private open = false;

  private toggleOpen() {
    this.open = !this.open;
  }
}
</script>

<style scoped>
  .header {
    border: 0 solid #e0e0e0;
    border-bottom-width: 1px;
    padding: 4px;
  }

  .header:hover {
    background: #2c2c2c;
    border-radius: 4px 4px 0 0;
    cursor: pointer;
  }

  .panel {
    margin-bottom: 10px;
  }
</style>
