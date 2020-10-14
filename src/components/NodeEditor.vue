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
import MathNode from '../nodes/model/Dense';

@Component
export default class NodeEditor extends Vue {
  editor = new Editor();

  optionPlugin = new OptionPlugin();

  viewPlugin = new ViewPlugin();

  created() {
    this.editor.use(this.optionPlugin);
    this.editor.use(this.viewPlugin);
    this.editor.registerNodeType('Dense', MathNode);
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

.node-editor{
  width: 500px;
  height: 500px;
}
</style>
