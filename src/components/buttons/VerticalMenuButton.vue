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

  .selected {
    background: var(--blue);
  }
</style>
