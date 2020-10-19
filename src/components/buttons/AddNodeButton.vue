<template>
  <div class="node-button" @click="addNode">{{name}}</div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component({})
export default class AddNodeButton extends Vue {
  @Prop({ required: true }) readonly node!: string;
  @Prop() readonly name!: string;

  private addNode() {
    const { editor } = this.$store.state;
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
  }

  .node-button:hover {
    background: #1c1c1c;
    cursor: pointer;
  }
</style>
