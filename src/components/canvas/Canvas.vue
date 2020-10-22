<template>
  <div class="canvas h-100">
    <baklava-editor :plugin="viewPlugin"></baklava-editor>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Editor } from '@baklavajs/core';
import CustomNode from '@/baklava/CustomNode.vue';
import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Engine } from '@baklavajs/plugin-engine';
import { traverseUiToIr } from '@/app/ir/traversals';

@Component
export default class Canvas extends Vue {
  @Prop({ required: true }) readonly editor!: Editor;
  @Prop({ required: true }) readonly abstractCanvas!: AbstractCanvas;

  optionPlugin = this.abstractCanvas.optionPlugin;
  viewPlugin = this.abstractCanvas.viewPlugin;
  engine = new Engine(true);

  created() {
    this.editor.use(this.optionPlugin);
    this.editor.use(this.viewPlugin);
    this.editor.use(this.engine);
    this.viewPlugin.components.node = CustomNode;

    this.abstractCanvas.registerOptions();
    this.abstractCanvas.registerNodes(this.editor);

    this.engine.events.calculated.addListener(this, (r) => {
      console.log('Something changed!');
      const state = this.editor.save();
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
