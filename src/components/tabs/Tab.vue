<template>
  <div v-if="visible" class="tab">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import Scrollable from '@/components/wrappers/Scrollable.vue';

@Component({
  components: { Scrollable },
})
export default class Tab extends Vue {
  @Prop({ required: true }) readonly name!: string;
  private visible = false;

  public setVisible(value: boolean) {
    this.visible = value;
  }
}
</script>

<style scoped>
  .tab {
    height: 100%;
    overflow: hidden;
  }
</style>
