<template>
  <div class="node-editor h-100">
    <baklava-editor :plugin="viewPlugin"></baklava-editor>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Editor } from '@baklavajs/core';
import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import { Layers, Nodes } from '@/nodes/model/Types';

// Importing the nodes
import Conv2D from '@/nodes/model/conv/Conv2D';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dense from '@/nodes/model/linear/Dense';
import Flatten from '@/nodes/model/reshape/Flatten';
import Dropout from '@/nodes/model/regularization/Dropout';

@Component
export default class NodeEditor extends Vue {
  editor = new Editor();

  optionPlugin = new OptionPlugin();

  viewPlugin = new ViewPlugin();

  created() {
    this.editor.use(this.optionPlugin);
    this.editor.use(this.viewPlugin);

    // Model Layer Nodes
    this.editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
    this.editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
    this.editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
    this.editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
    this.editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
  }
}
</script>

<style scoped>
</style>
