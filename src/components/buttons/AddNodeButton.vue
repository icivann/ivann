<template>
  <div class="node-button" @click="onClick" draggable="true" @dragend="dragEnd" :id="name">
    <div class="icon">
      <slot/>
    </div>
    <div class="name" :style="'font-size: ' + fontSize + 'em'">{{ name }}</div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { uniqueTextInput } from '@/inputs/prompt';
import EditorManager from '@/EditorManager';

@Component
export default class AddNodeButton extends Vue {
  @Prop({ required: true }) readonly node!: string;
  @Prop() readonly name!: string;
  @Prop() readonly options?: unknown;
  @Prop() readonly names?: Set<string>;

  private editorManager = EditorManager.getInstance();
  private fontSize = 1.0;

  private dragEnd = (event: DragEvent) => {
    if (this.editorManager.canDrop) {
      this.editorManager.enableDrop(false);
      // TODO: When different Node Widths are implemented, centre node on cursor.
      this.addNode(event.pageX - 160, event.pageY - 60);
    }
  };

  created() {
    let factor = (this.name.length - 10) / 3;
    if (factor > 0) {
      if (factor > 3) factor = 3;
      this.fontSize -= 0.18 * factor;
    }
  }

  private onClick() {
    this.addNode(window.innerWidth / 3, window.innerHeight / 3);
  }

  private addNode(x: number, y: number) {
    let name: string | null = null;
    if (this.names) {
      name = uniqueTextInput(this.names, 'Please enter a unique name for the IO');
      if (name === null) return;
    }

    const { editor } = this.$store.getters.currEditorModel;
    const NodeType = editor.nodeTypes.get(this.node);

    if (NodeType === undefined) {
      console.error(`Undefined Node Type: ${this.node}`);
    } else {
      const node = editor.addNode(new NodeType(this.options));

      // Set position (and name) of newly created node
      const { scaling, panning } = EditorManager.getInstance().viewPlugin;
      const { x: xPanning, y: yPanning } = panning;

      node.position.x = (x / scaling) - xPanning;
      node.position.y = (y / scaling) - yPanning;
      if (name) node.name = name;
    }
  }
}
</script>

<style scoped>
  .node-button {
    background: #202020;
    border-radius: 8px;
    text-align: center;
    color: #e0e0e0;
    font-size: initial;
    margin: 13px;
    border: 1px solid var(--grey);
    transition-duration: 0.1s;
    position: relative;
  }

  .node-button:hover {
    background: #1c1c1c;
    cursor: pointer;
    border-color: var(--foreground);
    transition-duration: 0.1s;
  }

  /*.icon {*/
  /*  height: 4em;*/
  /*  width: 4em;*/
  /*  margin: 1em auto;*/
  /*}*/

  /*.icon * {*/
  /*  max-height: 100%;*/
  /*  max-width: 100%;*/
  /*}*/

  .name {
    bottom: 0.3em;
    position: absolute;
    margin-left: auto;
    width: 100%;
  }
</style>
