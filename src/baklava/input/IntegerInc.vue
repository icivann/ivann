<template>
  <div class="box"
       @click="increment"
       @contextmenu="decrement">{{value}}
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class IntegerInc extends Vue {
  @Prop() value!: any;

  @Prop() index!: number;

  @Prop({ type: String }) name!: string;

  increment() {
    this.$emit('increment', this.value + 1, this.index);
  }

  decrement() {
    this.$emit('increment', this.value - 1, this.index);
  }
}
</script>

<style scoped>
  .box {
    margin: 0 0 0 5px;
    padding: 0 1em 0 0.3em;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
  }
</style>
