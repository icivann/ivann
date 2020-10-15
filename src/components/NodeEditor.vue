<template>
  <div>
    This is the Node editor
    <div class="node-editor">
      <baklava-editor :plugin="viewPlugin"></baklava-editor>
    </div>
    after baklava
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Editor } from '@baklavajs/core';
import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';

// Importing the nodes
import Conv2D from '@/nodes/model/conv/Conv2D';
import MaxPooling2D from '@/nodes/model/pool/MaxPooling2D';
import Dense from '@/nodes/model/linear/Dense';
import OptionTestNode from '@/nodes/OptionTestNode';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';

@Component
export default class NodeEditor extends Vue {
  editor = new Editor();

  optionPlugin = new OptionPlugin();

  viewPlugin = new ViewPlugin();

  created() {
    this.editor.use(this.optionPlugin);
    this.editor.use(this.viewPlugin);

    // Model Layer Nodes
    this.editor.registerNodeType('Dense', Dense);
    this.editor.registerNodeType('Conv2D', Conv2D, 'Convolution Layers');
    this.editor.registerNodeType('MaxPooling2D', MaxPooling2D, 'Pooling Layers');

    this.editor.registerNodeType('OptionTestNode', OptionTestNode);
    this.viewPlugin.registerOption('VectorOption', Vector);
    this.viewPlugin.registerOption('IntegerOption', Integer);
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

.node-editor{
  width: 100%;
  height: 500px;
}
</style>
