<template>
  <div class="canvas h-100" @dragover.prevent @dragenter="enableAddNode"
       @dragleave="disableAddNode" @drop="addNode">
    <baklava-editor :plugin="viewPlugin" :key="editorModel.id.toString()"/>
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
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import { EditorModel } from '@/store/editors/types';
import istateToGraph from '@/app/ir/istateToGraph';
import { saveEditor } from '@/file/EditorAsJson';
import EditorManager from '@/EditorManager';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
  @Prop({ required: true }) readonly engine!: Engine;
  @Prop({ required: true }) readonly editorModel!: EditorModel;

  private depthCounter = 0;
  private editorManager = EditorManager.getInstance();

  @Watch('editorModel')
  onEditorChange(newEditorModel: EditorModel) {
    newEditorModel.editor.use(this.viewPlugin);
    newEditorModel.editor.use(this.engine);
  }

  private addNode() {
    this.depthCounter -= 1;
    /* Node added by AddNodeButton */
  }

  private enableAddNode() {
    /* Keep track of accidentally dragged over other nodes. */
    this.depthCounter += 1;
    if (this.depthCounter === 1) {
      this.editorManager.enableDrop(true);
    }
  }

  private disableAddNode() {
    this.depthCounter -= 1;
    /* Don't disable if mouse over other node. */
    if (this.depthCounter === 0) {
      this.editorManager.enableDrop(false);
    }
  }

  created(): void {
    this.editorModel.editor.use(this.viewPlugin);
    this.editorModel.editor.use(this.engine);

    this.engine.events.calculated.addListener(this, () => {
      console.log('Something changed!');

      // Building IR
      const currEditorSave = saveEditor(this.editorModel);
      istateToGraph(currEditorSave.state);
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
