<template>
  <div v-on:click="onClick">
    <div v-if="up">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="16">
        <line x1="1" y1="8" x2="7" y2="1" fill="none" stroke="#ececec" stroke-width="2"/>
        <line x1="11.8" y1="8" x2="5.8" y2="1" fill="none" stroke="#ececec" stroke-width="2"/>
      </svg>
    </div>
    <div v-else>
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="16">
        <line x1="1" y1="1" x2="7" y2="8" fill="none" stroke="#ececec" stroke-width="2"/>
        <line x1="11.8" y1="1" x2="5.8" y2="8" fill="none" stroke="#ececec" stroke-width="2"/>
      </svg>
    </div>
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
