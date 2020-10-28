<template>
  <div
    class="button"
    :class="isSelected ? 'selected' : ''"
    @click="onClick">
    {{label}}
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class VerticalMenuButton extends Vue {
  @Prop({ required: true }) readonly label!: string;
  @Prop({ required: true }) readonly onClick!: () => void;
  @Prop({ required: true }) readonly isSelected!: boolean;
}
</script>

<style scoped>
  .button {
    background: #202020;
    text-align: center;
    color: #e0e0e0;
    font-size: 14px;
    padding: 10px;
  }

  .button:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    border-left-color: var(--blue);
    cursor: pointer;
  }

  .selected {
    color: var(--blue);
  }
</style>
