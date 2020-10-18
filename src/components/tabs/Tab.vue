<template>
  <div v-if="visible" class="">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import TabInterface from '@/components/tabs/TabInterface';

@Component
export default class Tab extends Vue implements TabInterface {
  @Prop() readonly name!: string;
  private visible = false;

  public setVisible(value: boolean) {
    this.visible = value;
  }
}
</script>
