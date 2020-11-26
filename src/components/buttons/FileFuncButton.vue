<template>
  <div
    class="func-button"
    :class="selected && 'selected'"
    @click="$emit('click')"
  >
    <div
      class="header"
      v-if="header !== undefined"
      v-text="header"
    />
    <div class="body">
      <slot/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class FileFuncButton extends Vue {
  @Prop({ required: false }) readonly header?: string;
  @Prop({ default: false }) readonly selected?: boolean;
}
</script>

<style scoped>
  .func-button {
    background: #202020;
    border-radius: 8px;
    color: var(--foreground);
    margin-bottom: 1em;
    border: 1px solid var(--grey);
    transition-duration: 0.1s;
    position: relative;
    padding: 0.5em 1em;
    font-family: monospace;
    font-size: 0.8em;
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

  .body {
    color: var(--grey);
  }

  .selected {
    color: var(--blue);
    border: 1px solid var(--blue);
  }

  .selected:hover {
    border: 1px solid #1B67E0;
  }
</style>
