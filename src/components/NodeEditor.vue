<template>
  <div class="node-editor h-100">
    <button @click="save"> Export Model </button>
    <baklava-editor :plugin="viewPlugin"></baklava-editor>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Editor } from '@baklavajs/core';
import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import { Engine } from '@baklavajs/plugin-engine';
import { Layers, Nodes } from '@/nodes/model/Types';

// Importing the nodes
import InModel from '@/nodes/model/InModel';
import OutModel from '@/nodes/model/OutModel';
import Conv2D from '@/nodes/model/conv/Conv2D';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dense from '@/nodes/model/linear/Dense';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';
import Dropdown from '@/baklava/options/Dropdown.vue';
import Flatten from '@/nodes/model/reshape/Flatten';
import Dropout from '@/nodes/model/regularization/Dropout';
import { FileSaver } from 'file-saver-typescript';
import { generateKeras } from '@/app/generators/keras/kerasGenerator';
import GraphNode from '@/app/ir/GraphNode';

@Component
export default class NodeEditor extends Vue {
  editor = new Editor();

  optionPlugin = new OptionPlugin();

  viewPlugin = new ViewPlugin();
  engine = new Engine(false);

  created() {
    this.editor.use(this.optionPlugin);
    this.editor.use(this.viewPlugin);
    this.editor.use(this.engine);

    // Model Layer Nodes
    this.viewPlugin.registerOption('VectorOption', Vector);
    this.viewPlugin.registerOption('IntegerOption', Integer);
    this.viewPlugin.registerOption('DropdownOption', Dropdown);
    this.editor.registerNodeType(Nodes.InModel, InModel, Layers.IO);
    this.editor.registerNodeType(Nodes.OutModel, OutModel, Layers.IO);
    this.editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
    this.editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
    this.editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
    this.editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
    this.editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
  }

  async save() {
    const r = await this.engine.calculate() as Map<any, any>;
    const data = r.entries().next().value;

    let model = data[1] as any[];
    model = model.filter((e) => e instanceof GraphNode);
    const code = generateKeras(model);

    const fileSaver: FileSaver = new FileSaver();
    fileSaver.responseData = code;
    fileSaver.strFileName = 'model.py';
    fileSaver.strMimeType = 'text/plain';
    fileSaver.initSaveFile();
    console.log('saved..');
  }
}
</script>

<style scoped>
</style>
