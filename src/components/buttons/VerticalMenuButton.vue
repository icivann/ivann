<template>
  <div :id="'button ' + (isSelected ? 'selected' : '')" @click="onClick">
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
  #button {
    background: #202020;
    border-radius: 8px;
    text-align: center;
    color: #e0e0e0;
    font-size: 14px;
    margin: 5px;
    border: 1px solid var(--grey);
    transition-duration: 0.1s;
    position: relative;
  }

  #button.selected {
    background: purple;
  }
</style>
