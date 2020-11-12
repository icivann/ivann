<template>
  <div
    class="button"
    :class="!disabled ? primary && 'primary' : 'disabled'"
    @click="!disabled ? $emit('click') : ''"
  >
    {{text}}
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class UIButton extends Vue {
  @Prop({ required: true }) text!: string;
  @Prop({ default: false }) primary?: boolean;
  @Prop({ default: false }) disabled?: boolean;
}
</script>

<style scoped lang="scss">
.button {
  background: var(--foreground);
  border-radius: 4px;
  padding: 0 1rem 2px 1rem;
  color: #303030;
  cursor: pointer;
  user-select: none;
  text-align: center;
  white-space: nowrap;
  width: fit-content;
  transition-duration: 0.1s;

  &:hover {
    background: #e0e0e0;
    transition-duration: 0.1s;
  }

  &:active {
    background: #dadada;
  }
}

.primary {
  background: var(--blue);
  color: var(--foreground);

  &:hover {
    background: #1B67E0;
  }

  &:active {
    background: #1860D6;
  }
}

.disabled {
  background: var(--grey);

  &:hover {
    background: var(--grey);
  }

  &:active {
    background: var(--grey);
  }
}
</style>
