<template>
  <div class="canvas h-100">
    <baklava-editor :plugin="viewPlugin" :key="editorModel.name"></baklava-editor>
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
import { EditorModel } from '@/store/Types';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
  @Prop({ required: true }) readonly editorModel!: EditorModel;

  @Watch('editorModel')
  onEditorChange(editorModel: EditorModel) {
    editorModel.editor.use(this.viewPlugin);
  }

  created(): void {
    this.editorModel.editor.use(this.viewPlugin);
  }
}
</script>

<style scoped>
</style>
