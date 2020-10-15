<template>
  <div class="box d-sm-flex">
    {{value}}
    <div>
      <div @click="increment">up</div>
      <div @click="decrement">down</div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class IntegerInc extends Vue {
  @Prop() value!: number;

  @Prop() index!: number;

  increment() {
    this.$emit('value-change', this.value + 1, this.index);
  }

  decrement() {
    this.$emit('value-change', this.value - 1, this.index);
  }
}
</script>

<style scoped>
  .box {
    margin: 0 0 0 5px;
    padding: 0 0em 0 0.3em;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
  }
</style>
