<template>
  <div class="canvas h-100">
    <baklava-editor :plugin="viewPlugin" :key="editorModel.id.toString()"></baklava-editor>
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
import { EditorModel, EditorModels } from '@/store/editors/types';
import istateToGraph from '@/app/ir/istateToGraph';
import { saveEditor, saveEditors } from '@/file/EditorAsJson';

@Component
export default class Canvas extends Vue {
  @Getter('allEditorModels') editorModels!: EditorModels;
  @Prop({ required: true }) readonly viewPlugin!: ViewPlugin;
  @Prop({ required: true }) readonly editorModel!: EditorModel;
  @Mutation('setUnsaved') setUnsaved!: (model: EditorModel) => void;

  public engine = new Engine(true);

  @Watch('editorModel')
  onEditorChange(editorModel: EditorModel) {
    editorModel.editor.use(this.viewPlugin);
    editorModel.editor.use(this.engine);
  }

  created(): void {
    this.editorModel.editor.use(this.viewPlugin);
    this.editorModel.editor.use(this.engine);

    this.engine.events.calculated.addListener(this, () => {
      console.log('Something changed!');
      this.setUnsaved(this.editorModel);
      // Auto-Saving
      const {
        overviewEditor,
        modelEditors,
        dataEditors,
        trainEditors,
      } = this.editorModels;

      const editorsSaved = {
        overviewEditor: saveEditor(overviewEditor),
        modelEditors: saveEditors(modelEditors),
        dataEditors: saveEditors(dataEditors),
        trainEditors: saveEditors(trainEditors),
      };
      this.$cookies.set('unsaved', editorsSaved);

      // Building IR
      const state = this.editorModel.editor.save();
      istateToGraph(state);
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
