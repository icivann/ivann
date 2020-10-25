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
import { Engine } from '@baklavajs/plugin-engine';
import { traverseUiToIr } from '@/app/ir/traversals';

import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import { EditorModel } from '@/store/editors/types';

@Component
export default class Canvas extends Vue {
    @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
    @Prop({ required: true }) readonly editorModel!: EditorModel;

    engine = new Engine(true);

  @Watch('editorModel')
    onEditorChange(editorModel: EditorModel) {
      editorModel.editor.use(this.viewPlugin);
    }

  created(): void {
    this.editorModel.editor.use(this.viewPlugin);
    this.editorModel.editor.use(this.engine);

    this.engine.events.calculated.addListener(this, (r) => {
      console.log('Something changed!');
      const state = this.editorModel.editor.save();
      traverseUiToIr(state);
    });
  }
}
</script>

<style>
  /* Used inside baklava-editor. */
  .node-editor {
    background-color: #242424;
    background-image: linear-gradient(#2c2c2c 1px, transparent 1px),
      linear-gradient(90deg, #2c2c2c 1px, transparent 1px);
    background-size: 10px 50px, 50px 50px, 10px 10px, 10px 10px;
  }
</style>
