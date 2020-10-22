<template>
  <div class="canvas h-100">
    <baklava-editor :plugin="viewPlugin"></baklava-editor>
  </div>
</template>

<script lang="ts">
import {
  Component,
  Prop,
  Vue,
  Watch,
} from 'vue-property-decorator';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;

  @Watch('viewPlugin')
  onViewPluginChange(viewPlugin: ViewPlugin) {
    console.log('fired');
    this.$store.getters.currEditor.use(viewPlugin);
  }

  created(): void {
    this.$store.getters.currEditor.use(this.viewPlugin);
  }
}
</script>

<style scoped>
</style>
