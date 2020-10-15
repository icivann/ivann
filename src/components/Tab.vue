<template>
  <div v-if="visible" class="">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component({})
export default class Tab extends Vue {
  @Prop() name!: string;
  visible = false;

  setVisible(value: boolean) {
    this.visible = value;
  }
}
</script>
