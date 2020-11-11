<template>
  <div class="func-button" :class="selected && 'selected'" @click="$emit('click')">
    <div class="header">{{ header }}</div>
    <div class="body">
      <slot/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class FileFuncButton extends Vue {
  @Prop({ required: true }) readonly header!: string;
  @Prop({ default: false }) readonly selected?: boolean;
}
</script>

<style scoped>
  .func-button {
    background: #202020;
    border-radius: 8px;
    color: var(--foreground);
    font-size: initial;
    margin-bottom: 1em;
    border: 1px solid var(--grey);
    transition-duration: 0.1s;
    position: relative;
    padding: 0.5em 1em;
  }

  .func-button:hover {
    background: #1c1c1c;
    cursor: pointer;
    border-color: var(--foreground);
    transition-duration: 0.1s;
  }

  .header {
    font-weight: bolder;
  }

  .selected {
    color: var(--blue);
    border: 1px solid var(--blue);
  }

  .selected:hover {
    border: 1px solid #1B67E0;
  }
</style>
