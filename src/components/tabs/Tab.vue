<template>
  <div v-if="visible" class="tab" :class="padded && 'padded'">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class Tab extends Vue {
  @Prop({ required: true }) readonly name!: string;
  @Prop({ default: true }) readonly padded?: boolean;
  private visible = false;

  public setVisible(value: boolean) {
    this.visible = value;
  }
}
</script>

<style scoped>
  .padded {
    margin: 1em;
  }

  .tab {
    height: 100%;
  }
</style>
