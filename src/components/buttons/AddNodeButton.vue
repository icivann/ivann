<template>
  <div class="node-button" @click="addNode">
    <div class="icon">
      <slot/>
    </div>
    <div class="name" :style="'font-size: ' + fontSize + 'em'">{{name}}</div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component({})
export default class AddNodeButton extends Vue {
  @Prop({ required: true }) readonly node!: string;
  @Prop() readonly name!: string;

  private fontSize = 1.0;

  created() {
    let factor = (this.name.length - 10) / 3;
    if (factor > 0) {
      if (factor > 3) factor = 3;
      this.fontSize -= 0.15 * factor;
    }
  }

  private addNode() {
    const { editor } = this.$store.getters.currEditorModel;
    const NodeType = editor.nodeTypes.get(this.node);

    if (NodeType === undefined) {
      console.error(`Undefined Node Type: ${this.node}`);
    } else {
      editor.addNode(new NodeType());
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
