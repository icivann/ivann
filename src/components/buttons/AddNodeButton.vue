<template>
  <div @click="addNode">{{name}}</div>
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

</style>
