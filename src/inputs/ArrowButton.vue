<template>
  <div v-on:click="onClick">
    <div v-if="up">▲</div>
    <div v-else>▼</div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class ArrowButton extends Vue {
  @Prop({ default: true }) initialUp!: boolean;

  private up = this.initialUp;

  onClick() {
    this.up = !this.up;
    this.$emit('arrow-button-clicked');
  }
}
</script>

<style>
</style>
