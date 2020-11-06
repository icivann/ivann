<template>
  <div class="canvas h-100">
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
import { Getter, Mutation } from 'vuex-class';
import { Engine } from '@baklavajs/plugin-engine';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import { EditorModel } from '@/store/editors/types';
import istateToGraph from '@/app/ir/istateToGraph';
import { EditorSave, saveEditor } from '@/file/EditorAsJson';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
  @Prop({ required: true }) readonly engine!: Engine;
  @Prop({ required: true }) readonly editorModel!: EditorModel;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Mutation('updateNodeInOverview') readonly updateNodeInOverview!: (cEditor: EditorModel) => void;

  @Watch('editorModel')
  onEditorChange(newEditorModel: EditorModel, oldEditorModel: EditorModel) {
    newEditorModel.editor.use(this.viewPlugin);
    newEditorModel.editor.use(this.engine);

    // Save oldEditorModel as periodic save may not have captured last changes
    const oldEditorSaved: EditorSave = saveEditor(oldEditorModel);
    const overviewEditorSave: EditorSave = saveEditor(this.overviewEditor);
    this.$cookies.set(`unsaved-editor-${oldEditorModel.name}`, oldEditorSaved);
    this.$cookies.set('unsaved-editor-Overview', overviewEditorSave);
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
