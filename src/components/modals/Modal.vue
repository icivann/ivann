<template>
  <div class="ivann-modal" v-if="value" @click="$emit('input', false)">
    <div class="modal-window" @click.stop>
      <div v-if="header !== undefined" class="header">{{ header }}</div>
      <slot/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class Modal extends Vue {
  @Prop({ required: true }) value!: boolean;
  @Prop() header?: string;
}
</script>

<style scoped>
.ivann-modal {
  color: var(--foreground);
  background-color: #00000060;
  position: absolute;
  z-index: 10;
  width: 100vw;
  height: 100vh;
  justify-content: center;
  align-items: center;
  display: flex;
}

.modal-window {
  background-color: var(--background);
  padding: 0.75rem;
  border: 1px var(--grey) solid;
  width: 20%;
  border-radius: 6px;
  user-select: none;
}

.header {
  margin-bottom: 1rem;
  text-align: center;
  font-size: smaller;
}
</style>
