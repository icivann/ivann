<template>
  <div class="canvas h-100">
    <baklava-editor :plugin="viewPlugin" :key="editor"></baklava-editor>
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
import { Editor } from '@baklavajs/core';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
  @Prop({ required: true }) readonly editor!: Editor;

  @Watch('editor')
  onEditorChange(editor: Editor, oldEditor: Editor) {
    if (editor !== oldEditor) {
      console.log('changed');
    }
    console.log('fired');
    editor.use(this.viewPlugin);
  }

  created(): void {
    this.editor.use(this.viewPlugin);
  }
}
</script>

<style scoped>
</style>
