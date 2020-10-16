<template>
  <div v-if="visible" class="">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class Tab extends Vue {
  @Prop() readonly name!: string;
  private visible = false;

  public setVisible(value: boolean) {
    this.visible = value;
  }
}
</script>
